from flask import Blueprint
from controllers import predict_controller
from extensions import limiter

predict_bp = Blueprint('predict', __name__)


@predict_bp.route('/api/predict', methods=['POST'], strict_slashes=False)
@limiter.limit("10 per minute")
def predict():
    """Deteksi Penyakit dari Gambar.
    ---
    tags:
      - Prediksi
    summary: Upload gambar daun → prediksi penyakit
    description: >
      Endpoint utama sistem. Upload gambar daun tanaman (JPG/PNG/WEBP),
      model AI akan menganalisis dan mengembalikan prediksi penyakit
      beserta confidence score (top-3) dan informasi penyakit lengkap.


      **Rate limit:** 10 request per menit per IP address.


      **Catatan:** Jika model belum ditraining, sistem akan berjalan dalam
      mode mock dan mengembalikan hasil simulasi.
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: Gambar daun tanaman (JPG, PNG, WEBP, BMP). Maksimum 10 MB.
    responses:
      200:
        description: Prediksi berhasil dilakukan
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: true
            predicted_class:
              type: string
              example: Tomato_healthy
            confidence:
              type: number
              format: float
              example: 0.9823
            confidence_percent:
              type: number
              example: 98.23
            disease_info:
              type: object
              properties:
                name_id:
                  type: string
                  example: Tomat Sehat
                plant:
                  type: string
                  example: Tomat
                status:
                  type: string
                  enum: [Sehat, Sakit]
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
                color:
                  type: string
            top_3:
              type: array
              description: Top-3 prediksi dengan confidence score
              items:
                type: object
                properties:
                  class:
                    type: string
                  confidence:
                    type: number
                  confidence_percent:
                    type: number
                  name_id:
                    type: string
                  plant:
                    type: string
                  status:
                    type: string
                  color:
                    type: string
            inference_time_ms:
              type: number
              example: 245.3
            model_mode:
              type: string
              enum: [real, mock]
              example: mock
      400:
        description: Request tidak valid (tidak ada file / format salah / file terlalu besar)
        schema:
          type: object
          properties:
            error:
              type: string
              example: Tidak ada file gambar dalam request
      500:
        description: Kesalahan internal server
        schema:
          type: object
          properties:
            error:
              type: string
    """
    from app import DISEASE_INFO
    return predict_controller.predict(DISEASE_INFO)
