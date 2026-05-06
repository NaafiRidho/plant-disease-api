import os
import json
import time
from flask import Flask, jsonify
from flask_cors import CORS
from flasgger import Swagger

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
        {"name": "Status",   "description": "Health check & server info"},
        {"name": "Penyakit", "description": "Informasi kelas penyakit tanaman"},
        {"name": "Prediksi", "description": "Deteksi penyakit dari gambar"},
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

# CORS: izinkan request origin dari Vercel dan Localhost
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "https://plant-disease-frontend-seven.vercel.app"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# ─────────────────────────────────────────────────────────────────────────────
# DATA & KONFIGURASI GLOBAL
# ─────────────────────────────────────────────────────────────────────────────

DISEASE_INFO_PATH = os.path.join(os.path.dirname(__file__), 'data', 'disease_info.json')
with open(DISEASE_INFO_PATH, 'r', encoding='utf-8') as f:
    DISEASE_INFO = json.load(f)

START_TIME = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# REGISTER BLUEPRINTS (Routes)
# ─────────────────────────────────────────────────────────────────────────────

from routes.status_routes import status_bp
from routes.disease_routes import disease_bp
from routes.predict_routes import predict_bp

app.register_blueprint(status_bp)
app.register_blueprint(disease_bp)
app.register_blueprint(predict_bp)

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
    from utils.model_utils import load_model

    print("=" * 60)
    print("  Sistem Pendeteksi Penyakit Tanaman - Flask Backend")
    print("=" * 60)

    model_loaded = load_model()

    if model_loaded:
        print("[OK] Model berhasil dimuat - Mode: REAL")
    else:
        print("[INFO] Model belum tersedia - Mode: MOCK")
        print("[INFO] Jalankan: cd ../ml-model && python train.py")

    print(f"[INFO] Disease info: {len(DISEASE_INFO)} kelas terdaftar")
    print(f"[INFO] Server berjalan di: http://localhost:5000")
    print("=" * 60)

    port = int(os.environ.get("PORT", 5000))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False  # Nonaktifkan auto-reload agar model tidak dimuat 2x
    )
