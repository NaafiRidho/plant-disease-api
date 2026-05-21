"""
predict_controller.py
──────────────────────
Controller untuk endpoint POST /api/predict.

Alur:
  1. Validasi file gambar (via image_validator)
  2. Jalankan inferensi ML (via model_utils)
  3. Bangun response JSON
  4. Simpan hasil ke tabel detection_histories (via detection_history_service)
  5. Return response — DB error tidak menggagalkan response prediksi
"""

import time

from flask import request

from utils.image_validator import validate_image_file, ImageValidationError
from utils.model_utils import predict_image
from utils.response_helper import error, server_error
from services import detection_history_service as history_svc
from flask import jsonify


def predict(disease_info: dict):
    """
    POST /api/predict

    Upload gambar daun tanaman → prediksi penyakit + simpan ke riwayat.

    Args:
        disease_info: Dict info penyakit yang di-load dari disease_info.json.

    Returns:
        JSON response dengan hasil prediksi.
    """

    # ── 1. Validasi file ──────────────────────────────────────────────────────
    if 'file' not in request.files:
        return error(
            "Tidak ada file gambar dalam request",
            400,
            details={"hint": "Kirim gambar dengan field name 'file'"},
        )

    file = request.files['file']

    try:
        image_bytes = validate_image_file(file)
    except ImageValidationError as exc:
        return error(str(exc), 400)

    # ── 2. Inferensi ML ───────────────────────────────────────────────────────
    try:
        start_time     = time.time()
        result         = predict_image(image_bytes)
        inference_time = round((time.time() - start_time) * 1000, 2)
    except ValueError as exc:
        return error(str(exc), 400, details={"type": "validation_error"})
    except RuntimeError as exc:
        return error(str(exc), 500, details={"type": "inference_error"})
    except Exception as exc:
        return server_error(f"Terjadi kesalahan saat inferensi: {exc}")

    # ── 3. Bangun response ────────────────────────────────────────────────────
    predicted_class = result['predicted_class']
    d_info          = disease_info.get(predicted_class, {})

    # Enriched top-3 dengan info penyakit
    top_3_enriched = []
    for item in result.get('top_3', []):
        info = disease_info.get(item['class'], {})
        top_3_enriched.append({
            **item,
            "name_id": info.get("name_id", item['class']),
            "plant":   info.get("plant",   "Unknown"),
            "status":  info.get("status",  "Unknown"),
            "color":   info.get("color",   "#6b7280"),
        })

    response_body = {
        "success":           True,
        "predicted_class":   predicted_class,
        "confidence":        result['confidence'],
        "confidence_percent": result['confidence_percent'],
        "disease_info": {
            "name_id":     d_info.get("name_id",     predicted_class),
            "plant":       d_info.get("plant",       "Unknown"),
            "status":      d_info.get("status",      "Unknown"),
            "description": d_info.get("description", ""),
            "symptoms":    d_info.get("symptoms",    []),
            "treatment":   d_info.get("treatment",   []),
            "severity":    d_info.get("severity",    "Unknown"),
            "color":       d_info.get("color",       "#6b7280"),
        },
        "top_3":             top_3_enriched,
        "inference_time_ms": inference_time,
        "model_mode":        "mock" if result.get('is_mock') else "real",
    }

    if result.get('is_mock'):
        response_body["mock_message"] = result.get('mock_message', '')

    # ── 4. Simpan ke detection_histories ─────────────────────────────────────
    try:
        is_healthy = d_info.get('status', '').lower() == 'sehat'

        record = history_svc.save_detection(
            filename        = file.filename or None,
            predicted_class = predicted_class,
            confidence      = result['confidence'],
            plant_type      = d_info.get('plant'),
            disease_name    = d_info.get('name_id'),
            is_healthy      = is_healthy,
            ip_address      = request.remote_addr,
            top_3           = top_3_enriched,
            user_id         = None,  # isi setelah auth diimplementasi
        )
        response_body["detection_id"] = record.id

    except Exception as db_exc:
        # DB error tidak menggagalkan response prediksi
        response_body["log_warning"] = (
            f"Prediksi berhasil, namun gagal menyimpan riwayat: {db_exc}"
        )

    # ── 5. Return response ────────────────────────────────────────────────────
    return jsonify(response_body), 200
