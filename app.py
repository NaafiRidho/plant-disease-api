import os
import json
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger
from utils.model_utils import load_model, predict_image, get_class_labels, is_model_loaded

# Inisialisasi Flask app
app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SWAGGER / FLASGGER CONFIG
# ─────────────────────────────────────────────────────────────────────────────

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "PlantScan AI — Plant Disease Detection API",
        "description": (
            "REST API untuk mendeteksi penyakit tanaman dari gambar daun menggunakan "
            "model CNN MobileNetV2 (Transfer Learning). Dataset: PlantVillage (15 kelas).\n\n"
            "**Tanaman yang didukung:** Paprika, Kentang, Tomat\n\n"
            "**Endpoint utama:** `POST /api/predict` — upload gambar → hasil prediksi + info penyakit"
        ),
        "version": "1.0.0",
        "contact": {
            "name": "PlantScan AI",
            "url": "https://plant-disease-frontend-seven.vercel.app/"
        },
        "license": {
            "name": "MIT"
        }
    },
    "host": "localhost:5000",
    "basePath": "/",
    "schemes": ["http"],
    "consumes": ["application/json", "multipart/form-data"],
    "produces": ["application/json"],
    "tags": [
        {"name": "Status",    "description": "Health check & server info"},
        {"name": "Penyakit",  "description": "Informasi kelas penyakit tanaman"},
        {"name": "Prediksi",  "description": "Deteksi penyakit dari gambar"},
    ]
}

swagger_config = {
    "headers": [],
    "specs": [{
        "endpoint": "apispec",
        "route": "/apispec.json",
        "rule_filter": lambda rule: True,
        "model_filter": lambda tag: True,
    }],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs",
    "title": "PlantScan AI — API Docs",
    "uiversion": 3,
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)

# CORS: izinkan request dari Next.js (localhost:3000) dan semua origin saat development
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Load disease info data
DISEASE_INFO_PATH = os.path.join(os.path.dirname(__file__), 'data', 'disease_info.json')
with open(DISEASE_INFO_PATH, 'r', encoding='utf-8') as f:
    DISEASE_INFO = json.load(f)

# Konfigurasi
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

# Server startup time
START_TIME = time.time()


def allowed_file(filename: str) -> bool:
    """Cek apakah ekstensi file diizinkan."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """API Info.
    ---
    tags:
      - Status
    summary: Informasi dasar API dan daftar endpoint
    responses:
      200:
        description: Informasi API berhasil diambil
        schema:
          type: object
          properties:
            message:
              type: string
              example: Sistem Pendeteksi Penyakit Tanaman API
            version:
              type: string
              example: "1.0.0"
            endpoints:
              type: object
    """
    return jsonify({
        "message": "Sistem Pendeteksi Penyakit Tanaman API",
        "version": "1.0.0",
        "docs": "http://localhost:5000/apidocs",
        "endpoints": {
            "health": "/api/health",
            "predict": "POST /api/predict",
            "classes": "/api/classes",
            "disease_info": "/api/disease/<class_name>"
        }
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Health Check.
    ---
    tags:
      - Status
    summary: Cek status server dan model ML
    description: >
      Mengembalikan status server, apakah model sudah dimuat,
      mode operasi (real/mock), dan uptime server.
    responses:
      200:
        description: Status server berhasil diambil
        schema:
          type: object
          properties:
            status:
              type: string
              example: ok
            model_loaded:
              type: boolean
              example: false
            model_mode:
              type: string
              enum: [real, mock]
              example: mock
            uptime_seconds:
              type: number
              example: 120.5
            supported_classes:
              type: integer
              example: 15
            message:
              type: string
              example: Server berjalan (model belum ditraining — mode mock aktif)
    """
    uptime = round(time.time() - START_TIME, 2)
    return jsonify({
        "status": "ok",
        "model_loaded": is_model_loaded(),
        "model_mode": "real" if is_model_loaded() else "mock",
        "uptime_seconds": uptime,
        "supported_classes": len(get_class_labels()),
        "message": "Server berjalan normal" if is_model_loaded() else "Server berjalan (model belum ditraining — mode mock aktif)"
    })


@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Daftar Semua Kelas Penyakit.
    ---
    tags:
      - Penyakit
    summary: Ambil semua 15 kelas penyakit tanaman
    description: >
      Mengembalikan daftar lengkap semua kelas penyakit yang didukung sistem,
      mencakup Paprika (2 kelas), Kentang (3 kelas), dan Tomat (10 kelas).
    responses:
      200:
        description: Daftar kelas berhasil diambil
        schema:
          type: object
          properties:
            total:
              type: integer
              example: 15
            classes:
              type: array
              items:
                type: object
                properties:
                  class_key:
                    type: string
                    example: Tomato_healthy
                  name_id:
                    type: string
                    example: Tomat Sehat
                  plant:
                    type: string
                    example: Tomat
                  status:
                    type: string
                    enum: [Sehat, Sakit]
                    example: Sehat
                  severity:
                    type: string
                    example: Tidak ada
                  color:
                    type: string
                    example: "#22c55e"
    """
    labels = get_class_labels()
    classes = []
    
    for label in labels:
        info = DISEASE_INFO.get(label, {})
        classes.append({
            "class_key": label,
            "name_id": info.get("name_id", label),
            "plant": info.get("plant", "Unknown"),
            "status": info.get("status", "Unknown"),
            "severity": info.get("severity", "Unknown"),
            "color": info.get("color", "#6b7280")
        })
    
    return jsonify({
        "total": len(classes),
        "classes": classes
    })


@app.route('/api/disease/<string:class_name>', methods=['GET'])
def get_disease_info(class_name: str):
    """Detail Informasi Penyakit.
    ---
    tags:
      - Penyakit
    summary: Ambil info lengkap satu kelas penyakit
    description: >
      Mengembalikan informasi detail penyakit tertentu, termasuk deskripsi,
      gejala, cara penanganan, dan tingkat keparahan.
      Gunakan nama kelas persis seperti di /api/classes (class_key).
    parameters:
      - name: class_name
        in: path
        type: string
        required: true
        description: Nama kelas penyakit (class_key)
        example: Tomato_healthy
    responses:
      200:
        description: Info penyakit berhasil diambil
        schema:
          type: object
          properties:
            class_key:
              type: string
              example: Tomato_healthy
            name_id:
              type: string
              example: Tomat Sehat
            plant:
              type: string
              example: Tomat
            status:
              type: string
              example: Sehat
            description:
              type: string
              example: Tanaman tomat dalam kondisi optimal.
            symptoms:
              type: array
              items:
                type: string
            treatment:
              type: array
              items:
                type: string
            severity:
              type: string
              example: Tidak ada
            color:
              type: string
              example: "#22c55e"
      404:
        description: Kelas penyakit tidak ditemukan
        schema:
          type: object
          properties:
            error:
              type: string
              example: Informasi penyakit 'xyz' tidak ditemukan
    """
    # Cari di disease info (case-insensitive partial match)
    info = DISEASE_INFO.get(class_name)
    
    if not info:
        # Coba cari partial match
        for key in DISEASE_INFO:
            if class_name.lower() in key.lower():
                info = DISEASE_INFO[key]
                break
    
    if not info:
        return jsonify({
            "error": f"Informasi penyakit '{class_name}' tidak ditemukan",
            "available_classes": list(DISEASE_INFO.keys())
        }), 404
    
    return jsonify({
        "class_key": class_name,
        **info
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Deteksi Penyakit dari Gambar.
    ---
    tags:
      - Prediksi
    summary: Upload gambar daun → prediksi penyakit
    description: >
      Endpoint utama sistem. Upload gambar daun tanaman (JPG/PNG/WEBP),
      model AI akan menganalisis dan mengembalikan prediksi penyakit
      beserta confidence score (top-3) dan informasi penyakit lengkap.


      **Catatan:** Jika model belum ditraining, sistem akan berjalan dalam
      mode mock dan mengembalikan hasil simulasi.
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: Gambar daun tanaman (JPG, PNG, WEBP, BMP). Maksimum 10 MB.
    responses:
      200:
        description: Prediksi berhasil dilakukan
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: true
            predicted_class:
              type: string
              example: Tomato_healthy
            confidence:
              type: number
              format: float
              example: 0.9823
            confidence_percent:
              type: number
              example: 98.23
            disease_info:
              type: object
              properties:
                name_id:
                  type: string
                  example: Tomat Sehat
                plant:
                  type: string
                  example: Tomat
                status:
                  type: string
                  enum: [Sehat, Sakit]
                description:
                  type: string
                symptoms:
                  type: array
                  items:
                    type: string
                treatment:
                  type: array
                  items:
                    type: string
                severity:
                  type: string
                color:
                  type: string
            top_3:
              type: array
              description: Top-3 prediksi dengan confidence score
              items:
                type: object
                properties:
                  class:
                    type: string
                  confidence:
                    type: number
                  confidence_percent:
                    type: number
                  name_id:
                    type: string
                  plant:
                    type: string
                  status:
                    type: string
                  color:
                    type: string
            inference_time_ms:
              type: number
              example: 245.3
            model_mode:
              type: string
              enum: [real, mock]
              example: mock
      400:
        description: Request tidak valid (tidak ada file / format salah / file terlalu besar)
        schema:
          type: object
          properties:
            error:
              type: string
              example: Tidak ada file gambar dalam request
      500:
        description: Kesalahan internal server
        schema:
          type: object
          properties:
            error:
              type: string
    """
    # Validasi: pastikan ada file dalam request
    if 'file' not in request.files:
        return jsonify({
            "error": "Tidak ada file gambar dalam request",
            "hint": "Kirim gambar dengan field name 'file'"
        }), 400
    
    file = request.files['file']
    
    # Validasi: pastikan file dipilih
    if file.filename == '':
        return jsonify({"error": "Tidak ada file yang dipilih"}), 400
    
    # Validasi: cek ekstensi file
    if not allowed_file(file.filename):
        return jsonify({
            "error": "Format file tidak didukung",
            "allowed_formats": list(ALLOWED_EXTENSIONS)
        }), 400
    
    # Validasi: cek ukuran file
    file.seek(0, 2)  # Seek ke akhir file
    file_size = file.tell()
    file.seek(0)     # Reset ke awal
    
    if file_size > MAX_FILE_SIZE:
        return jsonify({
            "error": f"Ukuran file terlalu besar. Maksimum {MAX_FILE_SIZE // (1024*1024)} MB",
            "file_size_mb": round(file_size / (1024*1024), 2)
        }), 400
    
    # Baca bytes gambar
    try:
        image_bytes = file.read()
    except Exception as e:
        return jsonify({"error": f"Gagal membaca file: {str(e)}"}), 500
    
    # Lakukan prediksi
    try:
        start_time = time.time()
        result = predict_image(image_bytes)
        inference_time = round((time.time() - start_time) * 1000, 2)  # ms
        
        # Ambil info penyakit dari prediksi terbaik
        predicted_class = result['predicted_class']
        disease_info = DISEASE_INFO.get(predicted_class, {})
        
        # Format top-3 dengan info penyakit
        top_3_with_info = []
        for item in result['top_3']:
            info = DISEASE_INFO.get(item['class'], {})
            top_3_with_info.append({
                **item,
                "name_id": info.get("name_id", item['class']),
                "plant": info.get("plant", "Unknown"),
                "status": info.get("status", "Unknown"),
                "color": info.get("color", "#6b7280")
            })
        
        response = {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": result['confidence'],
            "confidence_percent": result['confidence_percent'],
            "disease_info": {
                "name_id": disease_info.get("name_id", predicted_class),
                "plant": disease_info.get("plant", "Unknown"),
                "status": disease_info.get("status", "Unknown"),
                "description": disease_info.get("description", ""),
                "symptoms": disease_info.get("symptoms", []),
                "treatment": disease_info.get("treatment", []),
                "severity": disease_info.get("severity", "Unknown"),
                "color": disease_info.get("color", "#6b7280")
            },
            "top_3": top_3_with_info,
            "inference_time_ms": inference_time,
            "model_mode": "mock" if result.get('is_mock') else "real"
        }
        
        # Tambahkan pesan mock jika diperlukan
        if result.get('is_mock'):
            response["mock_message"] = result.get('mock_message', '')
        
        return jsonify(response)
    
    except ValueError as e:
        return jsonify({"error": str(e), "type": "validation_error"}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e), "type": "inference_error"}), 500
    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan internal: {str(e)}", "type": "server_error"}), 500


# ─────────────────────────────────────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint tidak ditemukan", "status": 404}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method tidak diizinkan", "status": 405}), 405


@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({"error": "File terlalu besar. Maksimum 10 MB", "status": 413}), 413


@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({"error": "Internal server error", "status": 500}), 500


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  Sistem Pendeteksi Penyakit Tanaman - Flask Backend")
    print("=" * 60)
    
    # Load model saat startup
    model_loaded = load_model()
    
    if model_loaded:
        print("[OK] Model berhasil dimuat - Mode: REAL")
    else:
        print("[INFO] Model belum tersedia - Mode: MOCK")
        print("[INFO] Jalankan: cd ../ml-model && python train.py")
    
    print(f"[INFO] Disease info: {len(DISEASE_INFO)} kelas terdaftar")
    print(f"[INFO] Server berjalan di: http://localhost:5000")
    print("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Nonaktifkan auto-reload agar model tidak dimuat 2x
    )
