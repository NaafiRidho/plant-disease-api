import os
import json
import time
from datetime import timedelta
from flask import Flask, jsonify
from flask_cors import CORS
from flasgger import Swagger
from dotenv import load_dotenv
from extensions import db, migrate, jwt, limiter

# Load environment variables dari file .env
load_dotenv()

# Inisialisasi Flask app
app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI DATABASE POSTGRESQL
# ─────────────────────────────────────────────────────────────────────────────

database_url = os.environ.get('DATABASE_URL')
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url or 'postgresql://postgres:password@localhost:5432/plantscan_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # nonaktifkan overhead tracking
app.config['SQLALCHEMY_ECHO'] = os.environ.get('FLASK_DEBUG', 'False') == 'True'  # log SQL di mode debug

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI JWT
# ─────────────────────────────────────────────────────────────────────────────

app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'super-secret-key-ganti-di-produksi')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(seconds=int(os.environ.get('JWT_ACCESS_TOKEN_EXPIRES', 3600)))
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=30)

# Inisialisasi ekstensi database
db.init_app(app)
migrate.init_app(app, db)
jwt.init_app(app)
limiter.init_app(app)

# ─────────────────────────────────────────────────────────────────────────────
# JWT BLOCKLIST — Cek token yang sudah di-logout (DB-persisted)
# ─────────────────────────────────────────────────────────────────────────────

@jwt.token_in_blocklist_loader
def check_if_token_in_blacklist(jwt_header, jwt_payload):
    """Cek JTI token di tabel token_blocklist. Persisten antar restart server."""
    from models.token_blocklist import TokenBlocklist
    jti = jwt_payload["jti"]
    return db.session.query(TokenBlocklist.id).filter_by(jti=jti).scalar() is not None

@jwt.revoked_token_loader
def revoked_token_callback(jwt_header, jwt_payload):
    from flask import jsonify
    return jsonify({"error": "Token telah dinonaktifkan. Silakan login kembali.", "status": 401}), 401

@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    from flask import jsonify
    return jsonify({"error": "Token sudah kadaluarsa. Silakan login kembali.", "status": 401}), 401

@jwt.invalid_token_loader
def invalid_token_callback(error):
    from flask import jsonify
    return jsonify({"error": "Token tidak valid.", "status": 401}), 401

@jwt.unauthorized_loader
def missing_token_callback(error):
    from flask import jsonify
    return jsonify({"error": "Token tidak ditemukan. Sertakan Authorization header.", "status": 401}), 401

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
        {"name": "Auth",             "description": "Login, logout, register & token management"},
        {"name": "Status",           "description": "Health check & server info"},
        {"name": "Penyakit",         "description": "Informasi kelas penyakit tanaman"},
        {"name": "Prediksi",         "description": "Deteksi penyakit dari gambar"},
        {"name": "Riwayat Deteksi",  "description": "Dashboard riwayat hasil deteksi penyakit"},
    ],
    "securityDefinitions": {
        "Bearer": {
            "type": "apiKey",
            "name": "Authorization",
            "in": "header",
            "description": "Masukkan token JWT dengan format: Bearer <token>"
        }
    }
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

# CORS: Izinkan request dari frontend (Vercel & Localhost)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",  # Izinkan semua origin sementara untuk testing
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Origin"]
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
from routes.auth_routes import auth_bp
from routes.detection_history_routes import detection_history_bp

app.register_blueprint(status_bp)
app.register_blueprint(disease_bp)
app.register_blueprint(predict_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(detection_history_bp)

# Import model agar Flask-Migrate dapat mendeteksi tabel
import models  # noqa: F401

# Jalankan migrasi database otomatis saat startup (sangat berguna untuk deploy Render)
from flask_migrate import upgrade as flask_db_upgrade
with app.app_context():
    try:
        print("[INFO] Menjalankan migrasi database otomatis...")
        flask_db_upgrade()
        print("[INFO] Migrasi database berhasil disinkronisasi.")
    except Exception as e:
        print(f"[WARNING] Gagal menjalankan migrasi otomatis: {e}")


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
    print(f"[INFO] Disease info: {len(DISEASE_INFO)} kelas terdaftar")

    # ── Pre-load model ML saat startup ───────────────────────────────────────
    # Menghindari cold start lambat pada request inferensi pertama
    print("[INFO] Memuat model ML...")
    from utils.model_utils import load_model
    with app.app_context():
        model_loaded = load_model()
        status_str = 'terlatih (real)' if model_loaded else 'mode mock aktif'
        print(f"[INFO] Model status: {status_str}")

    print(f"[INFO] Server berjalan di: http://localhost:5000")
    print("=" * 60)

    port = int(os.environ.get("PORT", 5000))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False
    )
