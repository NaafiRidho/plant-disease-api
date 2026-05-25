"""
detection_history_routes.py
────────────────────────────
Blueprint untuk endpoint riwayat deteksi penyakit tanaman.

Semua endpoint dilindungi JWT — hanya user yang login yang bisa akses.
Data yang dikembalikan hanya milik user yang sedang login.

Endpoints:
  GET /api/detection-histories          – list (milik user login)
  GET /api/detection-histories/stats    – statistik (milik user login)
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
