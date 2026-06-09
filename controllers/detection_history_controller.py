"""
detection_history_controller.py
────────────────────────────────
Controller untuk endpoint riwayat deteksi penyakit tanaman.

Semua endpoint memerlukan JWT (login). Data yang dikembalikan
hanya milik user yang sedang login (filter by user_id).
"""

from flask import request, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity

from services import detection_history_service as svc
from utils.response_helper import (
    success,
    success_list,
    not_found,
    server_error,
    build_pagination_meta,
)


def _current_user_id() -> int:
    """Ambil user_id (int) dari JWT identity."""
    return int(get_jwt_identity())


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/detection-histories
# ─────────────────────────────────────────────────────────────────────────────

@jwt_required()
def get_list():
    """
    Mengembalikan daftar riwayat deteksi milik user yang sedang login.

    Query params:
        page, per_page, search, plant_type, is_healthy,
        date_from, date_to, sort_by, order
    """
    try:
        user_id = _current_user_id()
        params  = request.args.to_dict()

        pagination, filters_applied = svc.get_detection_list(
            params, current_user_id=user_id
        )

        data = [item.to_list_dict() for item in pagination.items]
        meta = build_pagination_meta(pagination)

        return success_list(data, meta, filters_applied)

    except Exception as exc:
        return server_error(f"Gagal mengambil daftar riwayat deteksi: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/detection-histories/<id>
# ─────────────────────────────────────────────────────────────────────────────

@jwt_required()
def get_detail(detection_id: int):
    """
    Mengembalikan detail satu riwayat deteksi.
    Hanya bisa diakses jika record milik user yang sedang login.
    """
    try:
        user_id = _current_user_id()
        record  = svc.get_detection_by_id(detection_id)

        if record is None:
            return not_found("DetectionHistory", detection_id)

        # Pastikan record milik user yang login
        if record.user_id != user_id:
            return not_found("DetectionHistory", detection_id)

        data = record.to_detail_dict(disease_info=current_app.config['DISEASE_INFO'])

        return success(data)

    except Exception as exc:
        return server_error(f"Gagal mengambil detail riwayat deteksi: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/detection-histories/stats
# ─────────────────────────────────────────────────────────────────────────────

@jwt_required()
def get_stats():
    """
    Statistik riwayat deteksi milik user yang sedang login.
    """
    try:
        user_id = _current_user_id()
        stats   = svc.get_stats(current_user_id=user_id)
        return success(stats)

    except Exception as exc:
        return server_error(f"Gagal mengambil statistik: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/detection-histories/trend?period=weekly|monthly|yearly
# ─────────────────────────────────────────────────────────────────────────────

@jwt_required()
def get_trend():
    """
    Tren deteksi penyakit berdasarkan periode waktu untuk dashboard chart.

    Query param:
        period: 'weekly' (7 hari) | 'monthly' (12 bulan) | 'yearly' (5 tahun)
                Default: 'monthly'
    """
    try:
        user_id = _current_user_id()
        period  = request.args.get('period', 'monthly').strip().lower()

        if period not in ('weekly', 'monthly', 'yearly'):
            from utils.response_helper import error
            return error(
                "Parameter 'period' tidak valid. Gunakan: weekly, monthly, atau yearly.",
                400,
            )

        data = svc.get_trend(period=period, current_user_id=user_id)
        return success(data)

    except Exception as exc:
        return server_error(f"Gagal mengambil data tren: {exc}")
