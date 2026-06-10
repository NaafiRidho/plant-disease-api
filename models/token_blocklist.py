from datetime import datetime, timezone
from extensions import db


class TokenBlocklist(db.Model):
    """
    Tabel untuk menyimpan JWT ID (jti) yang sudah di-revoke (logout).

    Menggantikan in-memory set() yang tidak persisten antar restart server.
    Setiap token yang di-logout disimpan di sini dan dicek pada setiap request
    yang menggunakan JWT.
    """

    __tablename__ = 'token_blocklist'

    id         = db.Column(db.Integer, primary_key=True)
    jti        = db.Column(db.String(36), nullable=False, unique=True, index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return f'<TokenBlocklist jti={self.jti}>'

    @classmethod
    def cleanup(cls, max_age_days=30):
        """Hapus token yang sudah kadaluarsa (lebih lama dari max_age_days)."""
        from datetime import datetime, timezone, timedelta
        threshold = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        deleted = db.session.query(cls).filter(cls.created_at < threshold).delete()
        db.session.commit()
        return deleted

