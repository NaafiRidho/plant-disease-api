"""
detection_history_service.py
─────────────────────────────
Service layer untuk DetectionHistory.

Memisahkan business logic dari controller sehingga:
  - Controller hanya mengurus HTTP (parse request, return response)
  - Service mengurus query, filter, pagination, dan DB write
"""

import json
from datetime import datetime, timezone

from flask import request
from sqlalchemy import asc, desc, or_, func

from extensions import db
from models.detection_history import DetectionHistory

# ─────────────────────────────────────────────────────────────────────────────
# Konstanta
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PAGE     = 1
DEFAULT_PER_PAGE = 10
MAX_PER_PAGE     = 100

ALLOWED_SORT_FIELDS = {
    'id', 'filename', 'predicted_class', 'confidence',
    'plant_type', 'disease_name', 'is_healthy', 'created_at',
}


# ─────────────────────────────────────────────────────────────────────────────
# Write
# ─────────────────────────────────────────────────────────────────────────────

def save_detection(
    *,
    filename: str | None,
    predicted_class: str,
    confidence: float,
    plant_type: str | None,
    scientific_name: str | None,
    disease_name: str | None,
    severity: str | None,
    is_healthy: bool,
    ip_address: str | None,
    top_3: list | None = None,
    user_id: int | None = None,
    image_url: str | None = None,
) -> DetectionHistory:
    """
    Simpan satu record deteksi ke database.

    Args:
        filename:        Nama file yang di-upload.
        predicted_class: Kelas prediksi terbaik.
        confidence:      Confidence score (0.0–1.0).
        plant_type:      Jenis tanaman (dari disease_info.plant).
        scientific_name: Nama latin tanaman (dari disease_info.scientific_name).
        disease_name:    Nama penyakit (dari disease_info.name_id).
        severity:        Tingkat keparahan (dari disease_info.severity).
        is_healthy:      True jika tanaman sehat.
        ip_address:      IP address pengirim request.
        top_3:           List top-3 prediksi (akan di-encode ke JSON).
        user_id:         ID user jika sudah login (opsional).
        image_url:       URL Cloudinary dari gambar scan (opsional).

    Returns:
        Instance DetectionHistory yang sudah di-commit.

    Raises:
        Exception: Jika terjadi error saat commit ke DB.
    """
    record = DetectionHistory(
        filename        = filename,
        predicted_class = predicted_class,
        confidence      = confidence,
        plant_type      = plant_type,
        scientific_name = scientific_name,
        disease_name    = disease_name,
        severity        = severity,
        is_healthy      = is_healthy,
        ip_address      = ip_address,
        user_id         = user_id,
        image_url       = image_url,
    )
    record.top_3 = top_3 or []

    db.session.add(record)
    db.session.commit()
    return record


# ─────────────────────────────────────────────────────────────────────────────
# Read — list dengan filter, search, sort, pagination
# ─────────────────────────────────────────────────────────────────────────────

def get_detection_list(params: dict, current_user_id: int | None = None) -> tuple:
    """
    Query daftar DetectionHistory dengan filter, search, sort, dan pagination.

    Args:
        params:          Dict query params dari request.args.
        current_user_id: Jika diisi, hanya ambil data milik user tersebut.

    Returns:
        Tuple (pagination_object, filters_applied_dict)
    """
    query = DetectionHistory.query

    # ── Filter by user (jika login) ───────────────────────────────────────────
    if current_user_id is not None:
        query = query.filter(DetectionHistory.user_id == current_user_id)

    # ── Search ────────────────────────────────────────────────────────────────
    search = params.get('search', '').strip()
    if search:
        pattern = f'%{search}%'
        query = query.filter(
            or_(
                DetectionHistory.predicted_class.ilike(pattern),
                DetectionHistory.disease_name.ilike(pattern),
                DetectionHistory.filename.ilike(pattern),
                DetectionHistory.plant_type.ilike(pattern),
            )
        )

    # ── Filter: plant_type ────────────────────────────────────────────────────
    plant_type = params.get('plant_type', '').strip()
    if plant_type:
        query = query.filter(DetectionHistory.plant_type.ilike(plant_type))

    # ── Filter: is_healthy ────────────────────────────────────────────────────
    is_healthy_raw = params.get('is_healthy', '').strip().lower()
    if is_healthy_raw in ('true', '1'):
        query = query.filter(DetectionHistory.is_healthy == True)   # noqa: E712
    elif is_healthy_raw in ('false', '0'):
        query = query.filter(DetectionHistory.is_healthy == False)  # noqa: E712

    # ── Filter: date_from / date_to ───────────────────────────────────────────
    date_from_raw = params.get('date_from', '').strip()
    if date_from_raw:
        dt = _parse_date(date_from_raw, end_of_day=False)
        if dt:
            query = query.filter(DetectionHistory.created_at >= dt)

    date_to_raw = params.get('date_to', '').strip()
    if date_to_raw:
        dt = _parse_date(date_to_raw, end_of_day=True)
        if dt:
            query = query.filter(DetectionHistory.created_at <= dt)

    # ── Sorting ───────────────────────────────────────────────────────────────
    sort_by = params.get('sort_by', 'created_at').strip()
    if sort_by not in ALLOWED_SORT_FIELDS:
        sort_by = 'created_at'

    order = params.get('order', 'desc').strip().lower()
    sort_col = getattr(DetectionHistory, sort_by)
    query = query.order_by(asc(sort_col) if order == 'asc' else desc(sort_col))

    # ── Pagination ────────────────────────────────────────────────────────────
    page     = _parse_int(params.get('page'),     DEFAULT_PAGE)
    per_page = _parse_int(params.get('per_page'), DEFAULT_PER_PAGE)
    per_page = min(per_page, MAX_PER_PAGE)

    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    filters_applied = {
        'search':     search,
        'plant_type': plant_type,
        'is_healthy': is_healthy_raw,
        'date_from':  date_from_raw,
        'date_to':    date_to_raw,
        'sort_by':    sort_by,
        'order':      order,
    }

    return pagination, filters_applied


# ─────────────────────────────────────────────────────────────────────────────
# Read — detail by ID
# ─────────────────────────────────────────────────────────────────────────────

def get_detection_by_id(detection_id: int) -> DetectionHistory | None:
    """
    Ambil satu record DetectionHistory berdasarkan primary key.

    Returns:
        Instance DetectionHistory atau None jika tidak ditemukan.
    """
    return db.session.get(DetectionHistory, detection_id)


# ─────────────────────────────────────────────────────────────────────────────
# Read — tren chart (weekly / monthly / yearly)
# ─────────────────────────────────────────────────────────────────────────────

def get_trend(period: str, current_user_id: int | None = None) -> dict:
    """
    Hitung tren deteksi berdasarkan periode waktu.

    Args:
        period:          'weekly' | 'monthly' | 'yearly'
        current_user_id: Jika diisi, hanya data milik user tersebut.

    Returns:
        Dict berisi:
            - period        : periode yang dipakai
            - labels        : list label sumbu-X (string tanggal/minggu/bulan)
            - total         : list jumlah total deteksi per bucket
            - healthy       : list jumlah deteksi sehat per bucket
            - diseased      : list jumlah deteksi sakit per bucket
            - summary       : ringkasan angka keseluruhan dalam rentang
    """
    from sqlalchemy import func, case

    base_q = db.session.query(DetectionHistory)
    if current_user_id is not None:
        base_q = base_q.filter(DetectionHistory.user_id == current_user_id)

    now = datetime.now(timezone.utc)

    # ── Tentukan rentang waktu & fungsi grouping ──────────────────────────────
    if period == 'weekly':
        # 7 hari terakhir, bucket per hari
        start_dt   = now.replace(hour=0, minute=0, second=0, microsecond=0)
        from datetime import timedelta
        start_dt   = start_dt - timedelta(days=6)
        trunc_expr = func.date_trunc('day', DetectionHistory.created_at)
        label_fmt  = '%b %d'          # "Jun 09"
        buckets    = [start_dt + timedelta(days=i) for i in range(7)]

    elif period == 'monthly':
        # 12 bulan terakhir, bucket per bulan
        from datetime import timedelta
        # Awal bulan 11 bulan yang lalu
        start_dt   = (now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                      - timedelta(days=1))  # akhir bulan sebelumnya
        # Mundur ke 12 bulan lalu
        year  = now.year - (1 if now.month == 1 else 0)
        month = now.month - 1 if now.month > 1 else 12
        start_dt = now.replace(year=year if now.month > 1 else now.year - 1,
                               month=month, day=1,
                               hour=0, minute=0, second=0, microsecond=0)
        trunc_expr = func.date_trunc('month', DetectionHistory.created_at)
        label_fmt  = '%b %Y'          # "Jun 2026"
        # Buat 12 bucket bulanan
        buckets = []
        for i in range(12):
            m = (start_dt.month + i - 1) % 12 + 1
            y = start_dt.year + (start_dt.month + i - 1) // 12
            buckets.append(start_dt.replace(year=y, month=m, day=1))

    else:  # yearly
        # 5 tahun terakhir, bucket per tahun
        from datetime import timedelta
        start_dt   = now.replace(year=now.year - 4, month=1, day=1,
                                 hour=0, minute=0, second=0, microsecond=0)
        trunc_expr = func.date_trunc('year', DetectionHistory.created_at)
        label_fmt  = '%Y'             # "2026"
        buckets    = [start_dt.replace(year=start_dt.year + i) for i in range(5)]

    # ── Query agregat per bucket ──────────────────────────────────────────────
    rows = (
        base_q
        .filter(DetectionHistory.created_at >= start_dt)
        .with_entities(
            trunc_expr.label('bucket'),
            func.count(DetectionHistory.id).label('total'),
            func.sum(
                case((DetectionHistory.is_healthy == True, 1), else_=0)   # noqa: E712
            ).label('healthy'),
            func.sum(
                case((DetectionHistory.is_healthy == False, 1), else_=0)  # noqa: E712
            ).label('diseased'),
        )
        .group_by('bucket')
        .order_by('bucket')
        .all()
    )

    # Buat dict lookup: bucket_datetime → row
    row_map: dict = {}
    for row in rows:
        # date_trunc mengembalikan datetime — normalisasi ke awal periode
        if period == 'weekly':
            key = row.bucket.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'monthly':
            key = row.bucket.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            key = row.bucket.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        row_map[key] = row

    # ── Isi semua bucket (termasuk yang kosong = 0) ───────────────────────────
    labels:   list[str] = []
    totals:   list[int] = []
    healthys: list[int] = []
    diseaseds: list[int] = []

    for bucket_dt in buckets:
        labels.append(bucket_dt.strftime(label_fmt))
        row = row_map.get(bucket_dt)
        totals.append(int(row.total)   if row else 0)
        healthys.append(int(row.healthy)  if row else 0)
        diseaseds.append(int(row.diseased) if row else 0)

    return {
        'period':   period,
        'labels':   labels,
        'total':    totals,
        'healthy':  healthys,
        'diseased': diseaseds,
        'summary': {
            'total':    sum(totals),
            'healthy':  sum(healthys),
            'diseased': sum(diseaseds),
            'start_date': start_dt.strftime('%Y-%m-%d'),
            'end_date':   now.strftime('%Y-%m-%d'),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Read — statistik
# ─────────────────────────────────────────────────────────────────────────────

def get_stats(current_user_id: int | None = None) -> dict:
    """
    Hitung statistik ringkasan riwayat deteksi.

    Args:
        current_user_id: Jika diisi, statistik hanya untuk user tersebut.

    Returns:
        Dict berisi total, total_healthy, total_diseased,
        plant_breakdown, dan top_diseases.
    """
    base_query = DetectionHistory.query
    if current_user_id is not None:
        base_query = base_query.filter(DetectionHistory.user_id == current_user_id)

    total          = base_query.count()
    total_healthy  = base_query.filter(DetectionHistory.is_healthy == True).count()   # noqa: E712
    total_diseased = base_query.filter(DetectionHistory.is_healthy == False).count()  # noqa: E712

    # Breakdown per plant_type
    plant_q = (
        db.session.query(
            DetectionHistory.plant_type,
            func.count(DetectionHistory.id).label('count'),
        )
        .group_by(DetectionHistory.plant_type)
    )
    if current_user_id is not None:
        plant_q = plant_q.filter(DetectionHistory.user_id == current_user_id)

    plant_breakdown = [
        {'plant_type': row.plant_type or 'Unknown', 'count': row.count}
        for row in plant_q.all()
    ]

    # Top-5 penyakit terbanyak (hanya yang sakit)
    disease_q = (
        db.session.query(
            DetectionHistory.disease_name,
            func.count(DetectionHistory.id).label('count'),
        )
        .filter(DetectionHistory.is_healthy == False)   # noqa: E712
        .group_by(DetectionHistory.disease_name)
    )
    if current_user_id is not None:
        disease_q = disease_q.filter(DetectionHistory.user_id == current_user_id)

    disease_q = disease_q.order_by(desc(func.count(DetectionHistory.id))).limit(5)

    top_diseases = [
        {'disease_name': row.disease_name or 'Unknown', 'count': row.count}
        for row in disease_q.all()
    ]

    return {
        'total':           total,
        'total_healthy':   total_healthy,
        'total_diseased':  total_diseased,
        'plant_breakdown': plant_breakdown,
        'top_diseases':    top_diseases,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_int(value, default: int, min_val: int = 1) -> int:
    try:
        return max(min_val, int(value))
    except (TypeError, ValueError):
        return default


def _parse_date(raw: str, end_of_day: bool = False) -> datetime | None:
    """
    Parse string tanggal ke objek datetime.

    Mendukung format:
      - 'YYYY-MM-DD'
      - 'YYYY-MM-DDTHH:MM:SS'
      - 'YYYY-MM-DD HH:MM:SS'

    Args:
        raw:        String tanggal.
        end_of_day: Jika True dan hanya tanggal (tanpa waktu),
                    set waktu ke 23:59:59.

    Returns:
        datetime atau None jika format tidak valid.
    """
    formats = [
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(raw, fmt)
            # Jika hanya tanggal dan end_of_day=True, set ke akhir hari
            if end_of_day and fmt == '%Y-%m-%d':
                dt = dt.replace(hour=23, minute=59, second=59)
            return dt
        except ValueError:
            continue
    return None
