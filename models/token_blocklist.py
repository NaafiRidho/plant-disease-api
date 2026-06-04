from datetime import datetime
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
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f'<TokenBlocklist jti={self.jti}>'
