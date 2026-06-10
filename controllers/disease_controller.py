from flask import jsonify
from utils.model_utils import get_class_labels


def get_classes(disease_info: dict):
    """Mengembalikan daftar semua kelas penyakit yang didukung."""
    labels = get_class_labels()
    classes = []

    for label in labels:
        info = disease_info.get(label, {})
        classes.append({
            "class_key": label,
            "name_id": info.get("name_id", label),
            "scientific_name": info.get("scientific_name", "Unknown"),
            "description": info.get("description", ""),
            "plant": info.get("plant", "Unknown"),
            "status": info.get("status", "Unknown"),
            "severity": info.get("severity", "Unknown"),
            "color": info.get("color", "#6b7280")
        })

    return jsonify({
        "total": len(classes),
        "classes": classes
    })


def get_disease_info(class_name: str, disease_info: dict):
    """Mengembalikan informasi detail satu kelas penyakit."""
    info = disease_info.get(class_name)

    if not info:
        # Coba cari partial match (case-insensitive)
        for key in disease_info:
            if class_name.lower() in key.lower():
                info = disease_info[key]
                break

    if not info:
        return jsonify({
            "error": f"Informasi penyakit '{class_name}' tidak ditemukan",
            "available_classes": list(disease_info.keys())
        }), 404

    return jsonify({
        "class_key": class_name,
        **info
    })
