import os
import json
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.model_utils import load_model, predict_image, get_class_labels, is_model_loaded

# Inisialisasi Flask app
app = Flask(__name__)

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
    """Root endpoint — redirect info ke /api/health."""
    return jsonify({
        "message": "Sistem Pendeteksi Penyakit Tanaman API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "predict": "POST /api/predict",
            "classes": "/api/classes",
            "disease_info": "/api/disease/<class_name>"
        }
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
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
    """
    Endpoint untuk mendapatkan semua kelas penyakit beserta informasinya.
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
    """
    Endpoint untuk mendapatkan info detail satu penyakit.
    Gunakan underscore sebagai pemisah dalam URL.
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
    """
    Endpoint utama prediksi penyakit tanaman dari gambar.
    
    Request: multipart/form-data dengan field 'file' (gambar)
    Response: JSON dengan prediksi kelas dan confidence score
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
