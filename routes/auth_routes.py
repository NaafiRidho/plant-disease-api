from flask import Blueprint
from flask_jwt_extended import jwt_required
from controllers import auth_controller
from extensions import limiter

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')


@auth_bp.route('/register', methods=['POST'], strict_slashes=False)
@limiter.limit("5 per minute")
def register():
    """Registrasi pengguna baru.
    ---
    tags:
      - Auth
    summary: Mendaftarkan akun pengguna baru
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - username
            - email
            - password
          properties:
            username:
              type: string
              example: johndoe
            email:
              type: string
              example: john@example.com
            password:
              type: string
              example: rahasia123
    responses:
      201:
        description: Registrasi berhasil
        schema:
          type: object
          properties:
            message:
              type: string
              example: Registrasi berhasil
            status:
              type: integer
              example: 201
            user:
              type: object
      400:
        description: Input tidak valid
      409:
        description: Username atau email sudah terdaftar
    """
    return auth_controller.register()


@auth_bp.route('/login', methods=['POST'], strict_slashes=False)
@limiter.limit("10 per minute")
def login():
    """Login dan dapatkan JWT Token.
    ---
    tags:
      - Auth
    summary: Login pengguna dan mengembalikan access token + refresh token
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - password
          properties:
            username:
              type: string
              example: johndoe
              description: Bisa menggunakan username atau email
            email:
              type: string
              example: john@example.com
            password:
              type: string
              example: rahasia123
    responses:
      200:
        description: Login berhasil
        schema:
          type: object
          properties:
            message:
              type: string
              example: Login berhasil
            status:
              type: integer
              example: 200
            access_token:
              type: string
              example: eyJhbGci...
            refresh_token:
              type: string
              example: eyJhbGci...
            user:
              type: object
      401:
        description: Username/email atau password salah
      403:
        description: Akun dinonaktifkan
    """
    return auth_controller.login()


@auth_bp.route('/logout', methods=['POST'], strict_slashes=False)
@jwt_required()
def logout():
    """Logout dan nonaktifkan token.
    ---
    tags:
      - Auth
    summary: Logout pengguna — token dimasukkan ke blacklist
    security:
      - Bearer: []
    responses:
      200:
        description: Logout berhasil
        schema:
          type: object
          properties:
            message:
              type: string
              example: Logout berhasil. Token telah dinonaktifkan.
            status:
              type: integer
              example: 200
      401:
        description: Token tidak valid atau sudah expired
    """
    return auth_controller.logout()


@auth_bp.route('/refresh', methods=['POST'], strict_slashes=False)
@jwt_required(refresh=True)
def refresh():
    """Refresh Access Token.
    ---
    tags:
      - Auth
    summary: Memperbarui access token menggunakan refresh token
    security:
      - Bearer: []
    responses:
      200:
        description: Token berhasil diperbarui
        schema:
          type: object
          properties:
            message:
              type: string
              example: Token berhasil diperbarui
            status:
              type: integer
              example: 200
            access_token:
              type: string
              example: eyJhbGci...
      401:
        description: Refresh token tidak valid atau sudah expired
    """
    return auth_controller.refresh_token()


@auth_bp.route('/profile', methods=['GET'], strict_slashes=False)
@jwt_required()
def profile():
    """Profil pengguna yang sedang login.
    ---
    tags:
      - Auth
    summary: Mendapatkan data profil user berdasarkan JWT token
    security:
      - Bearer: []
    responses:
      200:
        description: Profil berhasil diambil
        schema:
          type: object
          properties:
            status:
              type: integer
              example: 200
            user:
              type: object
      401:
        description: Token tidak valid
      404:
        description: User tidak ditemukan
    """
    return auth_controller.get_profile()


@auth_bp.route('/profile', methods=['PUT'], strict_slashes=False)
@jwt_required()
def update_profile():
    """Update profil pengguna yang sedang login.
    ---
    tags:
      - Auth
    summary: Mengupdate profil pengguna (username, email, fullname, specialization, location, bio, password)
    security:
      - Bearer: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            username:
              type: string
            email:
              type: string
            fullname:
              type: string
            specialization:
              type: string
            location:
              type: string
            bio:
              type: string
            password:
              type: string
    responses:
      200:
        description: Profil berhasil diperbarui
      400:
        description: Request tidak valid
      409:
        description: Username atau email sudah digunakan
    """
    return auth_controller.update_profile()

