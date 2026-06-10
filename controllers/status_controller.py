import time
from flask import jsonify, request
from utils.model_utils import is_model_loaded, get_class_labels


def index():
    """Mengembalikan informasi dasar API dan daftar endpoint."""
    return jsonify({
        "message": "Sistem Pendeteksi Penyakit Tanaman API",
        "version": "1.0.0",
        "docs": f"{request.host_url.rstrip('/')}/apidocs",
        "endpoints": {
            "health": "/api/health",
            "predict": "POST /api/predict",
            "classes": "/api/classes",
            "disease_info": "/api/disease/<class_name>"
        }
    })


def health(start_time: float):
    """Mengembalikan status server dan model ML."""
    uptime = round(time.time() - start_time, 2)
    return jsonify({
        "status": "ok",
        "model_loaded": is_model_loaded(),
        "model_mode": "real" if is_model_loaded() else "mock",
        "uptime_seconds": uptime,
        "supported_classes": len(get_class_labels()),
        "message": (
            "Server berjalan normal"
            if is_model_loaded()
            else "Server berjalan (model belum ditraining — mode mock aktif)"
        )
    })
