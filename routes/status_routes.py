from flask import Blueprint, current_app
from controllers import status_controller

status_bp = Blueprint('status', __name__)


@status_bp.route('/')
def index():
    """API Info.
    ---
    tags:
      - Status
    summary: Informasi dasar API dan daftar endpoint
    responses:
      200:
        description: Informasi API berhasil diambil
        schema:
          type: object
          properties:
            message:
              type: string
              example: Sistem Pendeteksi Penyakit Tanaman API
            version:
              type: string
              example: "1.0.0"
            endpoints:
              type: object
    """
    return status_controller.index()


@status_bp.route('/api/health', methods=['GET'], strict_slashes=False)
def health():
    """Health Check.
    ---
    tags:
      - Status
    summary: Cek status server dan model ML
    description: >
      Mengembalikan status server, apakah model sudah dimuat,
      mode operasi (real/mock), dan uptime server.
    responses:
      200:
        description: Status server berhasil diambil
        schema:
          type: object
          properties:
            status:
              type: string
              example: ok
            model_loaded:
              type: boolean
              example: false
            model_mode:
              type: string
              enum: [real, mock]
              example: mock
            uptime_seconds:
              type: number
              example: 120.5
            supported_classes:
              type: integer
              example: 15
            message:
              type: string
              example: Server berjalan (model belum ditraining — mode mock aktif)
    """
    return status_controller.health(current_app.config['START_TIME'])
