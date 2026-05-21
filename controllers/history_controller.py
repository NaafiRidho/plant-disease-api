from flask import request, jsonify
from sqlalchemy import or_, asc, desc
from extensions import db
from models.prediction_log import PredictionLog

# ─────────────────────────────────────────────────────────────────────────────
# Konstanta
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_SORT_FIELDS = {
    'id', 'filename', 'predicted_class', 'confidence',
    'plant_type', 'disease_name', 'is_healthy', 'created_at'
}

DEFAULT_PAGE      = 1
DEFAULT_PER_PAGE  = 10
MAX_PER_PAGE      = 100


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _parse_int(value, default: int, min_val: int = 1) -> int:
    """Parse query param ke int dengan fallback ke default."""
    try:
        result = int(value)
        return max(min_val, result)
    except (TypeError, ValueError):
        return default


def _build_query():
    """
    Membangun SQLAlchemy query berdasarkan query-string request aktif.

    Query params yang didukung:
        search      – pencarian teks bebas di predicted_class, disease_name, filename
        plant_type  – filter exact (case-insensitive)
        is_healthy  – filter boolean: 'true' | 'false' | '1' | '0'
        date_from   – filter created_at >= (format ISO: YYYY-MM-DD atau YYYY-MM-DDTHH:MM:SS)
        date_to     – filter created_at <= (format ISO: YYYY-MM-DD atau YYYY-MM-DDTHH:MM:SS)
        sort_by     – kolom untuk sorting (default: created_at)
        order       – 'asc' | 'desc' (default: desc)
    """
    query = PredictionLog.query

    # ── Search ────────────────────────────────────────────────────────────────
    search = request.args.get('search', '').strip()
    if search:
        pattern = f'%{search}%'
        query = query.filter(
            or_(
                PredictionLog.predicted_class.ilike(pattern),
                PredictionLog.disease_name.ilike(pattern),
                PredictionLog.filename.ilike(pattern),
                PredictionLog.plant_type.ilike(pattern),
            )
        )

    # ── Filter: plant_type ────────────────────────────────────────────────────
    plant_type = request.args.get('plant_type', '').strip()
    if plant_type:
        query = query.filter(PredictionLog.plant_type.ilike(plant_type))

    # ── Filter: is_healthy ────────────────────────────────────────────────────
    is_healthy_raw = request.args.get('is_healthy', '').strip().lower()
    if is_healthy_raw in ('true', '1'):
        query = query.filter(PredictionLog.is_healthy == True)   # noqa: E712
    elif is_healthy_raw in ('false', '0'):
        query = query.filter(PredictionLog.is_healthy == False)  # noqa: E712

    # ── Filter: date_from / date_to ───────────────────────────────────────────
    from datetime import datetime

    date_from_raw = request.args.get('date_from', '').strip()
    if date_from_raw:
        try:
            # Coba parse dengan waktu, fallback ke tanggal saja
            try:
                date_from = datetime.fromisoformat(date_from_raw)
            except ValueError:
                date_from = datetime.strptime(date_from_raw, '%Y-%m-%d')
            query = query.filter(PredictionLog.created_at >= date_from)
        except ValueError:
            pass  # abaikan format tidak valid

    date_to_raw = request.args.get('date_to', '').strip()
    if date_to_raw:
        try:
            try:
                date_to = datetime.fromisoformat(date_to_raw)
            except ValueError:
                # Jika hanya tanggal, set ke akhir hari
                date_to = datetime.strptime(date_to_raw, '%Y-%m-%d').replace(
                    hour=23, minute=59, second=59
                )
            query = query.filter(PredictionLog.created_at <= date_to)
        except ValueError:
            pass

    # ── Sorting ───────────────────────────────────────────────────────────────
    sort_by = request.args.get('sort_by', 'created_at').strip()
    if sort_by not in ALLOWED_SORT_FIELDS:
        sort_by = 'created_at'

    order = request.args.get('order', 'desc').strip().lower()
    sort_col = getattr(PredictionLog, sort_by)
    query = query.order_by(asc(sort_col) if order == 'asc' else desc(sort_col))

    return query


# ─────────────────────────────────────────────────────────────────────────────
# Controller functions
# ─────────────────────────────────────────────────────────────────────────────

def get_histories():
    """
    GET /api/histories

    Mengembalikan daftar riwayat deteksi dengan pagination, search,
    sorting, dan filter.

    Query params:
        page        (int, default 1)
        per_page    (int, default 10, max 100)
        search      (str)  – cari di predicted_class, disease_name, filename, plant_type
        plant_type  (str)  – filter jenis tanaman
        is_healthy  (bool) – 'true'/'false'/'1'/'0'
        date_from   (str)  – ISO date/datetime, misal '2025-01-01'
        date_to     (str)  – ISO date/datetime, misal '2025-12-31'
        sort_by     (str)  – kolom sorting (default: created_at)
        order       (str)  – 'asc' | 'desc' (default: desc)
    """
    page     = _parse_int(request.args.get('page'),     DEFAULT_PAGE)
    per_page = _parse_int(request.args.get('per_page'), DEFAULT_PER_PAGE)
    per_page = min(per_page, MAX_PER_PAGE)

    try:
        query      = _build_query()
        pagination = query.paginate(page=page, per_page=per_page, error_out=False)
        items      = [item.to_dict() for item in pagination.items]

        return jsonify({
            "success": True,
            "data": items,
            "pagination": {
                "page":        pagination.page,
                "per_page":    pagination.per_page,
                "total":       pagination.total,
                "total_pages": pagination.pages,
                "has_next":    pagination.has_next,
                "has_prev":    pagination.has_prev,
                "next_page":   pagination.next_num,
                "prev_page":   pagination.prev_num,
            },
            "filters_applied": {
                "search":     request.args.get('search', ''),
                "plant_type": request.args.get('plant_type', ''),
                "is_healthy": request.args.get('is_healthy', ''),
                "date_from":  request.args.get('date_from', ''),
                "date_to":    request.args.get('date_to', ''),
                "sort_by":    request.args.get('sort_by', 'created_at'),
                "order":      request.args.get('order', 'desc'),
            }
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Gagal mengambil data riwayat: {str(e)}"
        }), 500


def get_history_detail(history_id: int):
    """
    GET /api/histories/<id>

    Mengembalikan detail satu riwayat deteksi berdasarkan ID.
    """
    try:
        log = PredictionLog.query.get(history_id)

        if log is None:
            return jsonify({
                "success": False,
                "error": f"Riwayat deteksi dengan id={history_id} tidak ditemukan"
            }), 404

        return jsonify({
            "success": True,
            "data": log.to_dict()
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Gagal mengambil detail riwayat: {str(e)}"
        }), 500


def get_history_stats():
    """
    GET /api/histories/stats

    Mengembalikan statistik ringkasan riwayat deteksi:
    total deteksi, jumlah sehat vs sakit, breakdown per plant_type.
    """
    try:
        from sqlalchemy import func

        total = PredictionLog.query.count()
        total_healthy   = PredictionLog.query.filter(PredictionLog.is_healthy == True).count()   # noqa: E712
        total_diseased  = PredictionLog.query.filter(PredictionLog.is_healthy == False).count()  # noqa: E712

        # Breakdown per plant_type
        plant_breakdown_rows = (
            db.session.query(
                PredictionLog.plant_type,
                func.count(PredictionLog.id).label('count')
            )
            .group_by(PredictionLog.plant_type)
            .all()
        )
        plant_breakdown = [
            {"plant_type": row.plant_type or "Unknown", "count": row.count}
            for row in plant_breakdown_rows
        ]

        # Top-5 penyakit terbanyak (hanya yang sakit)
        top_diseases_rows = (
            db.session.query(
                PredictionLog.disease_name,
                func.count(PredictionLog.id).label('count')
            )
            .filter(PredictionLog.is_healthy == False)   # noqa: E712
            .group_by(PredictionLog.disease_name)
            .order_by(desc(func.count(PredictionLog.id)))
            .limit(5)
            .all()
        )
        top_diseases = [
            {"disease_name": row.disease_name or "Unknown", "count": row.count}
            for row in top_diseases_rows
        ]

        return jsonify({
            "success": True,
            "stats": {
                "total":          total,
                "total_healthy":  total_healthy,
                "total_diseased": total_diseased,
                "plant_breakdown": plant_breakdown,
                "top_diseases":    top_diseases,
            }
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Gagal mengambil statistik: {str(e)}"
        }), 500
