from extensions import db
from datetime import datetime


class PredictionLog(db.Model):
    """
    Model untuk menyimpan riwayat prediksi penyakit tanaman.
    Setiap kali user mengupload gambar dan mendapat prediksi, data disimpan di sini.
    """

    __tablename__ = 'prediction_logs'

    id           = db.Column(db.Integer, primary_key=True)
    filename     = db.Column(db.String(255), nullable=True)          # nama file yang diupload
    predicted_class = db.Column(db.String(100), nullable=False)      # hasil kelas prediksi
    confidence   = db.Column(db.Float, nullable=False)               # confidence score (0.0 - 1.0)
    plant_type   = db.Column(db.String(50), nullable=True)           # jenis tanaman (Tomat, Kentang, dll)
    disease_name = db.Column(db.String(100), nullable=True)          # nama penyakit
    is_healthy   = db.Column(db.Boolean, default=False)              # apakah tanaman sehat?
    ip_address   = db.Column(db.String(50), nullable=True)           # IP pengirim request
    user_id      = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True) # ID user (opsional jika guest)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)  # waktu prediksi

    # Relasi ke model User
    user = db.relationship('User', backref=db.backref('predictions', lazy=True))

    def __repr__(self):
        return f'<PredictionLog id={self.id} class={self.predicted_class} conf={self.confidence:.2f}>'

    def to_dict(self):
        """Konversi model ke dictionary untuk JSON response."""
        return {
            'id':               self.id,
            'filename':         self.filename,
            'predicted_class':  self.predicted_class,
            'confidence':       round(self.confidence * 100, 2),  # dalam persen
            'plant_type':       self.plant_type,
            'disease_name':     self.disease_name,
            'is_healthy':       self.is_healthy,
            'ip_address':       self.ip_address,
            'user_id':          self.user_id,
            'created_at':       self.created_at.isoformat() if self.created_at else None,
        }
