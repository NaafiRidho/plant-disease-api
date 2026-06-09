"""
detection_history_routes.py
────────────────────────────
Blueprint untuk endpoint riwayat deteksi penyakit tanaman.

Semua endpoint dilindungi JWT — hanya user yang login yang bisa akses.
Data yang dikembalikan hanya milik user yang sedang login.

Endpoints:
  GET /api/detection-histories          – list (milik user login)
  GET /api/detection-histories/stats    – statistik (milik user login)
  GET /api/detection-histories/trend    – tren chart (weekly/monthly/yearly)
  GET /api/detection-histories/<id>     – detail (hanya jika milik user login)
"""

from flask import Blueprint
from flask_jwt_extended import jwt_required
from controllers import detection_history_controller as ctrl

detection_history_bp = Blueprint('detection_history', __name__)


# ─────────────────────────────────────────────────────────────────────────────
# Stats — didefinisikan SEBELUM /<int:id> agar Flask tidak salah routing
# ─────────────────────────────────────────────────────────────────────────────

@detection_history_bp.route(
    '/api/detection-histories/stats',
    methods=['GET'],
    strict_slashes=False,
)
@jwt_required()
def get_stats():
    """Statistik Riwayat Deteksi.
    ---
    tags:
      - Riwayat Deteksi
    summary: Statistik ringkasan riwayat deteksi milik user yang login
    security:
      - Bearer: []
    description: >
      Mengembalikan statistik agregat hanya untuk user yang sedang login:
      total deteksi, jumlah sehat vs sakit, breakdown per jenis tanaman,
      dan top-5 penyakit terbanyak.
    responses:
      200:
        description: Statistik berhasil diambil
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: true
            data:
              type: object
              properties:
                total:
                  type: integer
                  example: 120
                total_healthy:
                  type: integer
                  example: 45
                total_diseased:
                  type: integer
                  example: 75
                plant_breakdown:
                  type: array
                  items:
                    type: object
                    properties:
                      plant_type:
                        type: string
                        example: Tomat
                      count:
                        type: integer
                        example: 80
                top_diseases:
                  type: array
                  items:
                    type: object
                    properties:
                      disease_name:
                        type: string
                        example: Hawar Akhir Tomat
                      count:
                        type: integer
                        example: 30
      500:
        description: Kesalahan server
    """
    return ctrl.get_stats()


# ─────────────────────────────────────────────────────────────────────────────
# Trend chart — didefinisikan SEBELUM /<int:id> agar tidak salah routing
# ─────────────────────────────────────────────────────────────────────────────

@detection_history_bp.route(
    '/api/detection-histories/trend',
    methods=['GET'],
    strict_slashes=False,
)
@jwt_required()
def get_trend():
    """Tren Deteksi Penyakit (Dashboard Chart).
    ---
    tags:
      - Riwayat Deteksi
    summary: Data tren deteksi untuk chart dashboard dengan toggle periodik
    security:
      - Bearer: []
    description: >
      Mengembalikan data tren deteksi yang siap dipakai sebagai dataset chart
      (line chart / bar chart). Mendukung 3 mode periode yang bisa di-toggle
      oleh user di dashboard:


      - **weekly**  — 7 hari terakhir, bucket per hari (label: "Jun 09")

      - **monthly** — 12 bulan terakhir, bucket per bulan (label: "Jun 2026")

      - **yearly**  — 5 tahun terakhir, bucket per tahun (label: "2026")


      Setiap bucket berisi total deteksi, jumlah sehat, dan jumlah sakit.
      Bucket yang tidak ada datanya tetap dikembalikan dengan nilai 0
      sehingga chart tidak ada lubang.
    parameters:
      - name: period
        in: query
        type: string
        required: false
        default: monthly
        enum: [weekly, monthly, yearly]
        description: Periode waktu untuk agregasi tren
    responses:
      200:
        description: Data tren berhasil diambil
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: true
            data:
              type: object
              properties:
                period:
                  type: string
                  example: monthly
                  enum: [weekly, monthly, yearly]
                labels:
                  type: array
                  description: Label sumbu-X untuk chart (urut dari terlama ke terbaru)
                  items:
                    type: string
                  example: ["Jul 2025", "Aug 2025", "Sep 2025", "Oct 2025",
                            "Nov 2025", "Dec 2025", "Jan 2026", "Feb 2026",
                            "Mar 2026", "Apr 2026", "May 2026", "Jun 2026"]
                total:
                  type: array
                  description: Jumlah total deteksi per bucket
                  items:
                    type: integer
                  example: [0, 2, 5, 3, 0, 8, 12, 7, 4, 0, 15, 29]
                healthy:
                  type: array
                  description: Jumlah deteksi tanaman sehat per bucket
                  items:
                    type: integer
                  example: [0, 1, 2, 1, 0, 3, 5, 2, 2, 0, 6, 10]
                diseased:
                  type: array
                  description: Jumlah deteksi tanaman sakit per bucket
                  items:
                    type: integer
                  example: [0, 1, 3, 2, 0, 5, 7, 5, 2, 0, 9, 19]
                summary:
                  type: object
                  description: Ringkasan total dalam rentang periode tersebut
                  properties:
                    total:
                      type: integer
                      example: 85
                    healthy:
                      type: integer
                      example: 32
                    diseased:
                      type: integer
                      example: 53
                    start_date:
                      type: string
                      example: "2025-07-01"
                    end_date:
                      type: string
                      example: "2026-06-09"
      400:
        description: Parameter period tidak valid
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: false
            error:
              type: string
              example: "Parameter 'period' tidak valid. Gunakan: weekly, monthly, atau yearly."
      500:
        description: Kesalahan server
    """
    return ctrl.get_trend()


# ─────────────────────────────────────────────────────────────────────────────
# List
# ─────────────────────────────────────────────────────────────────────────────

@detection_history_bp.route(
    '/api/detection-histories',
    methods=['GET'],
    strict_slashes=False,
)
@jwt_required()
def get_list():
    """Daftar Riwayat Deteksi Penyakit.
    ---
    tags:
      - Riwayat Deteksi
    summary: Ambil daftar riwayat deteksi milik user yang login
    security:
      - Bearer: []
    description: >
      Mengembalikan daftar riwayat deteksi **milik user yang sedang login**.
      Mendukung pagination, pencarian teks bebas, filter, dan sorting.
    parameters:
      - name: page
        in: query
        type: integer
        default: 1
        description: Nomor halaman
      - name: per_page
        in: query
        type: integer
        default: 10
        description: Jumlah item per halaman (maks 100)
      - name: search
        in: query
        type: string
        description: Kata kunci pencarian bebas
      - name: plant_type
        in: query
        type: string
        description: Filter jenis tanaman
        enum: [Tomat, Kentang, Paprika]
      - name: is_healthy
        in: query
        type: string
        description: Filter status kesehatan
        enum: ["true", "false"]
      - name: date_from
        in: query
        type: string
        description: Filter dari tanggal (YYYY-MM-DD)
        example: "2025-01-01"
      - name: date_to
        in: query
        type: string
        description: Filter sampai tanggal (YYYY-MM-DD)
        example: "2025-12-31"
      - name: sort_by
        in: query
        type: string
        default: created_at
        description: Kolom untuk sorting
        enum: [id, filename, predicted_class, confidence, plant_type, disease_name, is_healthy, created_at]
      - name: order
        in: query
        type: string
        default: desc
        description: Arah sorting
        enum: [asc, desc]
    responses:
      200:
        description: Daftar riwayat berhasil diambil
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: true
            data:
              type: array
              items:
                type: object
                properties:
                  id:
                    type: integer
                    example: 1
                  filename:
                    type: string
                    example: tomato.jpg
                  predicted_class:
                    type: string
                    example: Tomato_Late_blight
                  confidence:
                    type: number
                    example: 0.9823
                  confidence_value:
                    type: number
                    example: 98.23
                    description: Nilai 0–100 untuk lebar progress bar di UI
                  confidence_percent:
                    type: string
                    example: "98.23%"
                  plant_type:
                    type: string
                    example: Tomat
                  scientific_name:
                    type: string
                    example: Solanum lycopersicum
                    description: Nama latin tanaman untuk subtitle card UI
                  disease_name:
                    type: string
                    example: Hawar Akhir Tomat
                  severity:
                    type: string
                    example: Sangat Tinggi
                    enum: [Tidak ada, Sedang, Tinggi, Sangat Tinggi]
                  is_healthy:
                    type: boolean
                    example: false
                  display_status:
                    type: string
                    example: Health Alert
                    enum: [Flourishing, Health Alert, Under Treatment]
                    description: Label badge status untuk tampilan UI history card
                  status_color:
                    type: string
                    example: "#ef4444"
                    description: Warna hex untuk badge status di UI
                  image_url:
                    type: string
                    example: "https://res.cloudinary.com/..."
                    description: URL foto hasil scan dari Cloudinary
                  user_id:
                    type: integer
                    example: 1
                  scanned_at:
                    type: string
                    example: "May 21, 10:00"
                    description: Format tanggal ringkas untuk ditampilkan di UI card
                  created_at:
                    type: string
                    example: "2025-05-21 10:00:00"
            pagination:
              type: object
              properties:
                page:
                  type: integer
                  example: 1
                per_page:
                  type: integer
                  example: 10
                total:
                  type: integer
                  example: 100
                total_pages:
                  type: integer
                  example: 10
                has_next:
                  type: boolean
                  example: true
                has_prev:
                  type: boolean
                  example: false
                next_page:
                  type: integer
                  example: 2
                prev_page:
                  type: integer
                  example: null
            filters_applied:
              type: object
              description: Filter yang sedang aktif pada request ini
      500:
        description: Kesalahan server
    """
    return ctrl.get_list()


# ─────────────────────────────────────────────────────────────────────────────
# Detail
# ─────────────────────────────────────────────────────────────────────────────

@detection_history_bp.route(
    '/api/detection-histories/<int:detection_id>',
    methods=['GET'],
    strict_slashes=False,
)
@jwt_required()
def get_detail(detection_id: int):
    """Detail Riwayat Deteksi.
    ---
    tags:
      - Riwayat Deteksi
    summary: Ambil detail satu riwayat deteksi berdasarkan ID (hanya milik user login)
    security:
      - Bearer: []
    parameters:
      - name: detection_id
        in: path
        type: integer
        required: true
        description: ID riwayat deteksi
        example: 1
    responses:
      200:
        description: Detail riwayat berhasil diambil
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: true
            data:
              type: object
              properties:
                id:
                  type: integer
                  example: 1
                filename:
                  type: string
                  example: tomato.jpg
                predicted_class:
                  type: string
                  example: Tomato_Late_blight
                confidence:
                  type: number
                  example: 0.98
                confidence_percent:
                  type: string
                  example: "98.0%"
                plant_type:
                  type: string
                  example: Tomat
                disease_name:
                  type: string
                  example: Hawar Akhir Tomat
                is_healthy:
                  type: boolean
                  example: false
                ip_address:
                  type: string
                  example: 127.0.0.1
                user_id:
                  type: integer
                  example: null
                created_at:
                  type: string
                  example: "2025-05-21 10:00:00"
                disease_info:
                  type: object
                  properties:
                    name_id:
                      type: string
                      example: Hawar Akhir Tomat
                    plant:
                      type: string
                      example: Tomat
                    status:
                      type: string
                      example: Sakit
                    description:
                      type: string
                    symptoms:
                      type: array
                      items:
                        type: string
                    treatment:
                      type: array
                      items:
                        type: string
                    severity:
                      type: string
                      example: Sangat Tinggi
                    color:
                      type: string
                      example: "#dc2626"
                top_3:
                  type: array
                  description: Top-3 prediksi saat deteksi dilakukan
                  items:
                    type: object
      404:
        description: Riwayat tidak ditemukan
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: false
            error:
              type: string
              example: "DetectionHistory dengan id=99 tidak ditemukan"
      500:
        description: Kesalahan server
    """
    return ctrl.get_detail(detection_id)
