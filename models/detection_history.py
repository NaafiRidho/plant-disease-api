import json
from datetime import datetime
from extensions import db


class DetectionHistory(db.Model):
    """
    Tabel riwayat deteksi penyakit tanaman.
    Setiap prediksi yang berhasil akan disimpan di sini.
    """

    __tablename__ = 'detection_histories'

    id              = db.Column(db.Integer, primary_key=True, autoincrement=True)
    filename        = db.Column(db.String(255), nullable=True)
    predicted_class = db.Column(db.String(100), nullable=False)
    confidence      = db.Column(db.Float, nullable=False)           # raw 0.0–1.0
    plant_type      = db.Column(db.String(50), nullable=True)
    disease_name    = db.Column(db.String(100), nullable=True)
    is_healthy      = db.Column(db.Boolean, nullable=False, default=False)
    ip_address      = db.Column(db.String(50), nullable=True)
    top_3_json      = db.Column(db.Text, nullable=True)             # JSON string top-3 predictions
    user_id         = db.Column(
        db.Integer,
        db.ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
        index=True,
    )
    created_at      = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )

    # Relationship
    user = db.relationship('User', backref=db.backref('detection_histories', lazy='dynamic'))

    # ── Indexes untuk query yang sering dipakai ───────────────────────────────
    __table_args__ = (
        db.Index('ix_dh_plant_type',   'plant_type'),
        db.Index('ix_dh_is_healthy',   'is_healthy'),
        db.Index('ix_dh_created_at',   'created_at'),
    )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def confidence_percent(self) -> str:
        """Confidence dalam format persen string, misal '98.23%'."""
        return f"{round(self.confidence * 100, 2)}%"

    @property
    def top_3(self) -> list:
        """Decode top_3_json ke list Python."""
        if not self.top_3_json:
            return []
        try:
            return json.loads(self.top_3_json)
        except (json.JSONDecodeError, TypeError):
            return []

    @top_3.setter
    def top_3(self, value: list):
        """Encode list ke JSON string untuk disimpan."""
        self.top_3_json = json.dumps(value, ensure_ascii=False) if value else None

    # ── Serializers ───────────────────────────────────────────────────────────

    def to_list_dict(self) -> dict:
        """
        Serialisasi ringkas untuk endpoint list.
        Tidak menyertakan top_3 dan disease_info detail agar response ringan.
        """
        return {
            'id':                self.id,
            'filename':          self.filename,
            'predicted_class':   self.predicted_class,
            'confidence':        round(self.confidence, 4),
            'confidence_percent': self.confidence_percent,
            'plant_type':        self.plant_type,
            'disease_name':      self.disease_name,
            'is_healthy':        self.is_healthy,
            'ip_address':        self.ip_address,
            'user_id':           self.user_id,
            'created_at':        self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
        }

    def to_detail_dict(self, disease_info: dict | None = None) -> dict:
        """
        Serialisasi lengkap untuk endpoint detail.
        Menyertakan top_3 dan disease_info jika tersedia.
        """
        d_info = {}
        if disease_info:
            d_info = disease_info.get(self.predicted_class, {})

        return {
            'id':                self.id,
            'filename':          self.filename,
            'predicted_class':   self.predicted_class,
            'confidence':        round(self.confidence, 4),
            'confidence_percent': self.confidence_percent,
            'plant_type':        self.plant_type,
            'disease_name':      self.disease_name,
            'is_healthy':        self.is_healthy,
            'ip_address':        self.ip_address,
            'user_id':           self.user_id,
            'created_at':        self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'disease_info':      d_info or None,
            'top_3':             self.top_3,
        }

    def __repr__(self) -> str:
        return (
            f'<DetectionHistory id={self.id} '
            f'class={self.predicted_class} '
            f'conf={self.confidence:.4f}>'
        )
