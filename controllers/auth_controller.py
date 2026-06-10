import re
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
from models.token_blocklist import TokenBlocklist
from datetime import datetime, timezone


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

    # Validasi kekuatan password
    if not re.search(r'[A-Z]', password):
        return jsonify({"error": "Password harus mengandung minimal 1 huruf besar", "status": 400}), 400

    if not re.search(r'[a-z]', password):
        return jsonify({"error": "Password harus mengandung minimal 1 huruf kecil", "status": 400}), 400

    if not re.search(r'[0-9]', password):
        return jsonify({"error": "Password harus mengandung minimal 1 angka", "status": 400}), 400

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
    """Logout pengguna dengan menyimpan JTI token ke tabel blocklist di database."""
    jti = get_jwt()["jti"]

    # Simpan JTI ke database — persisten antar restart server
    try:
        record = TokenBlocklist(jti=jti)
        db.session.add(record)
        db.session.commit()
    except Exception:
        # Jika JTI sudah ada (duplicate), abaikan
        db.session.rollback()

    # Panggil cleanup token blocklist
    try:
        TokenBlocklist.cleanup()
    except Exception:
        pass

    return jsonify({
        "message": "Logout berhasil. Token telah dinonaktifkan.",
        "status": 200
    }), 200




@jwt_required(refresh=True)
def refresh_token():
    """Memperbarui access token menggunakan refresh token."""
    user_id = get_jwt_identity()
    user = db.session.get(User, int(user_id))  # SQLAlchemy 2.0 style

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
    user = db.session.get(User, int(user_id))  # SQLAlchemy 2.0 style

    if not user:
        return jsonify({"error": "User tidak ditemukan", "status": 404}), 404

    return jsonify({
        "status": 200,
        "user": user.to_dict()
    }), 200


@jwt_required()
def update_profile():
    """Mengupdate profil pengguna yang sedang login."""
    user_id = get_jwt_identity()
    user = db.session.get(User, int(user_id))

    if not user:
        return jsonify({"error": "User tidak ditemukan", "status": 404}), 404

    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body tidak boleh kosong", "status": 400}), 400

    # Validasi dan update field username
    if 'username' in data:
        username = data['username'].strip()
        if username and username != user.username:
            if User.query.filter_by(username=username).first():
                return jsonify({"error": "Username sudah digunakan", "status": 409}), 409
            user.username = username

    # Validasi dan update field email
    if 'email' in data:
        email = data['email'].strip().lower()
        if email and email != user.email:
            if User.query.filter_by(email=email).first():
                return jsonify({"error": "Email sudah terdaftar", "status": 409}), 409
            user.email = email

    # Update password jika ada
    if 'password' in data and data['password']:
        old_password = data.get('old_password')
        if not old_password:
            return jsonify({"error": "Password lama wajib diisi untuk mengubah password", "status": 400}), 400
        if not user.check_password(old_password):
            return jsonify({"error": "Password lama salah", "status": 400}), 400
            
        password = data['password']
        if len(password) < 6:
            return jsonify({"error": "Password minimal 6 karakter", "status": 400}), 400
        if not re.search(r'[A-Z]', password):
            return jsonify({"error": "Password harus mengandung minimal 1 huruf besar", "status": 400}), 400
        if not re.search(r'[a-z]', password):
            return jsonify({"error": "Password harus mengandung minimal 1 huruf kecil", "status": 400}), 400
        if not re.search(r'[0-9]', password):
            return jsonify({"error": "Password harus mengandung minimal 1 angka", "status": 400}), 400
        user.set_password(password)

    db.session.commit()

    return jsonify({
        "message": "Profil berhasil diperbarui",
        "status": 200,
        "user": user.to_dict()
    }), 200
