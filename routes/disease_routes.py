from flask import Blueprint
from controllers import disease_controller

disease_bp = Blueprint('disease', __name__)


@disease_bp.route('/api/classes', methods=['GET'])
def get_classes():
    """Daftar Semua Kelas Penyakit.
    ---
    tags:
      - Penyakit
    summary: Ambil semua 15 kelas penyakit tanaman
    description: >
      Mengembalikan daftar lengkap semua kelas penyakit yang didukung sistem,
      mencakup Paprika (2 kelas), Kentang (3 kelas), dan Tomat (10 kelas).
    responses:
      200:
        description: Daftar kelas berhasil diambil
        schema:
          type: object
          properties:
            total:
              type: integer
              example: 15
            classes:
              type: array
              items:
                type: object
                properties:
                  class_key:
                    type: string
                    example: Tomato_healthy
                  name_id:
                    type: string
                    example: Tomat Sehat
                  plant:
                    type: string
                    example: Tomat
                  status:
                    type: string
                    enum: [Sehat, Sakit]
                    example: Sehat
                  severity:
                    type: string
                    example: Tidak ada
                  color:
                    type: string
                    example: "#22c55e"
    """
    from app import DISEASE_INFO
    return disease_controller.get_classes(DISEASE_INFO)


@disease_bp.route('/api/disease/<string:class_name>', methods=['GET'])
def get_disease_info(class_name: str):
    """Detail Informasi Penyakit.
    ---
    tags:
      - Penyakit
    summary: Ambil info lengkap satu kelas penyakit
    description: >
      Mengembalikan informasi detail penyakit tertentu, termasuk deskripsi,
      gejala, cara penanganan, dan tingkat keparahan.
      Gunakan nama kelas persis seperti di /api/classes (class_key).
    parameters:
      - name: class_name
        in: path
        type: string
        required: true
        description: Nama kelas penyakit (class_key)
        example: Tomato_healthy
    responses:
      200:
        description: Info penyakit berhasil diambil
        schema:
          type: object
          properties:
            class_key:
              type: string
              example: Tomato_healthy
            name_id:
              type: string
              example: Tomat Sehat
            plant:
              type: string
              example: Tomat
            status:
              type: string
              example: Sehat
            description:
              type: string
              example: Tanaman tomat dalam kondisi optimal.
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
              example: Tidak ada
            color:
              type: string
              example: "#22c55e"
      404:
        description: Kelas penyakit tidak ditemukan
        schema:
          type: object
          properties:
            error:
              type: string
              example: Informasi penyakit 'xyz' tidak ditemukan
    """
    from app import DISEASE_INFO
    return disease_controller.get_disease_info(class_name, DISEASE_INFO)
