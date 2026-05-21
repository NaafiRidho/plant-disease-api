from flask import jsonify, request
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    jwt_required,
    get_jwt_identity,
    get_jwt
)
from extensions import db
from models.user import User
from datetime import datetime, timezone


# Set untuk menyimpan token yang sudah di-blacklist (logout)
# Catatan: Untuk produksi, gunakan Redis atau database
jwt_blacklist = set()


def register():
    """Mendaftarkan pengguna baru."""
    data = request.get_json()

    # Validasi input
    if not data:
        return jsonify({"error": "Request body tidak boleh kosong", "status": 400}), 400

    username = data.get('username', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not username or not email or not password:
        return jsonify({"error": "Username, email, dan password wajib diisi", "status": 400}), 400

    if len(password) < 6:
        return jsonify({"error": "Password minimal 6 karakter", "status": 400}), 400

    # Cek apakah username atau email sudah terdaftar
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username sudah digunakan", "status": 409}), 409

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email sudah terdaftar", "status": 409}), 409

    # Buat user baru
    user = User(username=username, email=email)
    user.set_password(password)

    db.session.add(user)
    db.session.commit()

    return jsonify({
        "message": "Registrasi berhasil",
        "status": 201,
        "user": user.to_dict()
    }), 201


def login():
    """Login pengguna dan mengembalikan JWT token."""
    data = request.get_json()

    if not data:
        return jsonify({"error": "Request body tidak boleh kosong", "status": 400}), 400

    identifier = data.get('username') or data.get('email', '')
    password = data.get('password', '')

    if not identifier or not password:
        return jsonify({"error": "Username/email dan password wajib diisi", "status": 400}), 400

    # Cari user berdasarkan username atau email
    user = User.query.filter(
        (User.username == identifier) | (User.email == identifier.lower())
    ).first()

    if not user or not user.check_password(password):
        return jsonify({"error": "Username/email atau password salah", "status": 401}), 401

    if not user.is_active:
        return jsonify({"error": "Akun Anda telah dinonaktifkan", "status": 403}), 403

    # Buat JWT access token dan refresh token
    access_token = create_access_token(
        identity=str(user.id),
        additional_claims={"role": user.role, "username": user.username}
    )
    refresh_token = create_refresh_token(identity=str(user.id))

    return jsonify({
        "message": "Login berhasil",
        "status": 200,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user": user.to_dict()
    }), 200


@jwt_required()
def logout():
    """Logout pengguna dengan memasukkan token ke blacklist."""
    jti = get_jwt()["jti"]  # JWT ID unik dari token saat ini
    jwt_blacklist.add(jti)

    return jsonify({
        "message": "Logout berhasil. Token telah dinonaktifkan.",
        "status": 200
    }), 200


@jwt_required(refresh=True)
def refresh_token():
    """Memperbarui access token menggunakan refresh token."""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)

    if not user or not user.is_active:
        return jsonify({"error": "User tidak ditemukan atau tidak aktif", "status": 404}), 404

    new_access_token = create_access_token(
        identity=str(user.id),
        additional_claims={"role": user.role, "username": user.username}
    )

    return jsonify({
        "message": "Token berhasil diperbarui",
        "status": 200,
        "access_token": new_access_token
    }), 200


@jwt_required()
def get_profile():
    """Mendapatkan data profil pengguna yang sedang login."""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)

    if not user:
        return jsonify({"error": "User tidak ditemukan", "status": 404}), 404

    return jsonify({
        "status": 200,
        "user": user.to_dict()
    }), 200
