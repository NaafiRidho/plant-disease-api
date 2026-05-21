"""
detection_history_controller.py
────────────────────────────────
Controller untuk endpoint riwayat deteksi penyakit tanaman.

Tanggung jawab controller:
  - Parse dan validasi query params dari HTTP request
  - Memanggil service layer untuk business logic
  - Membangun HTTP response menggunakan response_helper

Business logic (query, filter, DB) ada di:
  services/detection_history_service.py
"""

from flask import request

from services import detection_history_service as svc
from utils.response_helper import (
    success,
    success_list,
    not_found,
    server_error,
    build_pagination_meta,
)


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/detection-histories
# ─────────────────────────────────────────────────────────────────────────────

def get_list():
    """
    Mengembalikan daftar riwayat deteksi dengan pagination, search,
    filter, dan sorting.

    Query params yang didukung:
        page        (int, default 1)
        per_page    (int, default 10, max 100)
        search      (str)  – cari di predicted_class, disease_name, filename, plant_type
        plant_type  (str)  – filter jenis tanaman (case-insensitive)
        is_healthy  (str)  – 'true' | 'false' | '1' | '0'
        date_from   (str)  – ISO date: 'YYYY-MM-DD' atau 'YYYY-MM-DDTHH:MM:SS'
        date_to     (str)  – ISO date: 'YYYY-MM-DD' atau 'YYYY-MM-DDTHH:MM:SS'
        sort_by     (str)  – kolom sorting (default: created_at)
        order       (str)  – 'asc' | 'desc' (default: desc)
    """
    try:
        params = request.args.to_dict()
        pagination, filters_applied = svc.get_detection_list(params)

        data = [item.to_list_dict() for item in pagination.items]
        meta = build_pagination_meta(pagination)

        return success_list(data, meta, filters_applied)

    except Exception as exc:
        return server_error(f"Gagal mengambil daftar riwayat deteksi: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/detection-histories/<id>
# ─────────────────────────────────────────────────────────────────────────────

def get_detail(detection_id: int):
    """
    Mengembalikan detail satu riwayat deteksi berdasarkan ID.
    Menyertakan disease_info lengkap dan top_3 predictions.
    """
    try:
        record = svc.get_detection_by_id(detection_id)

        if record is None:
            return not_found("DetectionHistory", detection_id)

        # Ambil disease_info dari app context (di-load saat startup)
        from app import DISEASE_INFO
        data = record.to_detail_dict(disease_info=DISEASE_INFO)

        return success(data)

    except Exception as exc:
        return server_error(f"Gagal mengambil detail riwayat deteksi: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/detection-histories/stats
# ─────────────────────────────────────────────────────────────────────────────

def get_stats():
    """
    Mengembalikan statistik ringkasan riwayat deteksi:
    total, sehat vs sakit, breakdown per tanaman, top-5 penyakit.
    """
    try:
        stats = svc.get_stats()
        return success(stats)

    except Exception as exc:
        return server_error(f"Gagal mengambil statistik: {exc}")
