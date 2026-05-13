from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

# Inisialisasi ekstensi di sini, bukan di app.py
# Ini menghindari circular import saat model diimport dari banyak tempat
db = SQLAlchemy()
migrate = Migrate()
