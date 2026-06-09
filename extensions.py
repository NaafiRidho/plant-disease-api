import os
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Inisialisasi ekstensi di sini, bukan di app.py
# Ini menghindari circular import saat model diimport dari banyak tempat
db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()

# Rate limiter — storage backend bisa dikonfigurasi via environment variable.
# Development: default "memory://" (reset setiap restart server)
# Production:  set RATELIMIT_STORAGE_URI=redis://localhost:6379/0 untuk persistence
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=os.environ.get("RATELIMIT_STORAGE_URI", "memory://"),
)
