from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager

# Inisialisasi ekstensi di sini, bukan di app.py
# Ini menghindari circular import saat model diimport dari banyak tempat
db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()
