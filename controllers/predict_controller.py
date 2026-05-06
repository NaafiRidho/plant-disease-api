import time
from flask import request, jsonify
from utils.model_utils import predict_image

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}


def allowed_file(filename: str) -> bool:
    """Cek apakah ekstensi file diizinkan."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict(disease_info: dict):
    """Memproses upload gambar dan mengembalikan prediksi penyakit."""

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
    file.seek(0, 2)
    file_size = file.tell()
    file.seek(0)

    if file_size > MAX_FILE_SIZE:
        return jsonify({
            "error": f"Ukuran file terlalu besar. Maksimum {MAX_FILE_SIZE // (1024 * 1024)} MB",
            "file_size_mb": round(file_size / (1024 * 1024), 2)
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
        inference_time = round((time.time() - start_time) * 1000, 2)

        predicted_class = result['predicted_class']
        d_info = disease_info.get(predicted_class, {})

        # Format top-3 dengan info penyakit
        top_3_with_info = []
        for item in result['top_3']:
            info = disease_info.get(item['class'], {})
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
                "name_id": d_info.get("name_id", predicted_class),
                "plant": d_info.get("plant", "Unknown"),
                "status": d_info.get("status", "Unknown"),
                "description": d_info.get("description", ""),
                "symptoms": d_info.get("symptoms", []),
                "treatment": d_info.get("treatment", []),
                "severity": d_info.get("severity", "Unknown"),
                "color": d_info.get("color", "#6b7280")
            },
            "top_3": top_3_with_info,
            "inference_time_ms": inference_time,
            "model_mode": "mock" if result.get('is_mock') else "real"
        }

        if result.get('is_mock'):
            response["mock_message"] = result.get('mock_message', '')

        return jsonify(response)

    except ValueError as e:
        return jsonify({"error": str(e), "type": "validation_error"}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e), "type": "inference_error"}), 500
    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan internal: {str(e)}", "type": "server_error"}), 500
