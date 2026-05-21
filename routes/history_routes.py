from flask import Blueprint
from controllers import history_controller

history_bp = Blueprint('history', __name__)


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/histories/stats  (harus didefinisikan SEBELUM /<int:history_id>
#                            agar Flask tidak salah routing 'stats' sebagai int)
# ─────────────────────────────────────────────────────────────────────────────

@history_bp.route('/api/histories/stats', methods=['GET'], strict_slashes=False)
def get_history_stats():
    """Statistik Ringkasan Riwayat Deteksi.
    ---
    tags:
      - Riwayat Deteksi
    summary: Statistik total deteksi, sehat vs sakit, dan breakdown per tanaman
    description: >
      Mengembalikan ringkasan statistik dari seluruh riwayat deteksi:
      total deteksi, jumlah tanaman sehat, jumlah tanaman sakit,
      breakdown per jenis tanaman, dan top-5 penyakit terbanyak.
    responses:
      200:
        description: Statistik berhasil diambil
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: true
            stats:
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
                        example: Tomato Early Blight
                      count:
                        type: integer
                        example: 30
      500:
        description: Kesalahan server
    """
    return history_controller.get_history_stats()


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/histories
# ─────────────────────────────────────────────────────────────────────────────

@history_bp.route('/api/histories', methods=['GET'], strict_slashes=False)
def get_histories():
    """Daftar Riwayat Deteksi Penyakit.
    ---
    tags:
      - Riwayat Deteksi
    summary: Ambil daftar riwayat deteksi dengan pagination, search, filter, dan sorting
    description: >
      Mengembalikan daftar riwayat deteksi penyakit tanaman yang tersimpan di database.
      Mendukung pagination, pencarian teks bebas, filter berdasarkan jenis tanaman,
      status kesehatan, dan rentang tanggal, serta sorting multi-kolom.


      **Filter yang tersedia:**
      - `search` — cari di predicted_class, disease_name, filename, plant_type
      - `plant_type` — filter jenis tanaman (Tomat, Kentang, Paprika)
      - `is_healthy` — filter status: `true` / `false`
      - `date_from` — filter mulai tanggal (format: YYYY-MM-DD)
      - `date_to` — filter sampai tanggal (format: YYYY-MM-DD)


      **Sorting:** gunakan `sort_by` + `order` (asc/desc).
      Kolom yang bisa di-sort: id, filename, predicted_class, confidence,
      plant_type, disease_name, is_healthy, created_at.
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
        description: Jumlah item per halaman (maksimum 100)
      - name: search
        in: query
        type: string
        description: Kata kunci pencarian
      - name: plant_type
        in: query
        type: string
        description: Filter jenis tanaman (Tomat / Kentang / Paprika)
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
                    example: leaf_tomato.jpg
                  predicted_class:
                    type: string
                    example: Tomato_Early_blight
                  confidence:
                    type: number
                    example: 92.45
                  plant_type:
                    type: string
                    example: Tomat
                  disease_name:
                    type: string
                    example: Early Blight
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
                    example: "2025-05-20T10:30:00"
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
                  example: 120
                total_pages:
                  type: integer
                  example: 12
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
              description: Filter yang sedang aktif
      500:
        description: Kesalahan server
    """
    return history_controller.get_histories()


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/histories/<id>
# ─────────────────────────────────────────────────────────────────────────────

@history_bp.route('/api/histories/<int:history_id>', methods=['GET'], strict_slashes=False)
def get_history_detail(history_id: int):
    """Detail Riwayat Deteksi.
    ---
    tags:
      - Riwayat Deteksi
    summary: Ambil detail satu riwayat deteksi berdasarkan ID
    parameters:
      - name: history_id
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
                  example: leaf_tomato.jpg
                predicted_class:
                  type: string
                  example: Tomato_Early_blight
                confidence:
                  type: number
                  example: 92.45
                plant_type:
                  type: string
                  example: Tomat
                disease_name:
                  type: string
                  example: Early Blight
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
                  example: "2025-05-20T10:30:00"
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
              example: "Riwayat deteksi dengan id=99 tidak ditemukan"
      500:
        description: Kesalahan server
    """
    return history_controller.get_history_detail(history_id)
