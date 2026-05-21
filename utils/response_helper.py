"""
response_helper.py
──────────────────
Helper terpusat untuk membangun JSON response yang konsisten di seluruh API.

Semua endpoint harus menggunakan fungsi-fungsi ini agar format response
seragam dan mudah dikonsumsi oleh frontend.
"""

from flask import jsonify
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Success responses
# ─────────────────────────────────────────────────────────────────────────────

def success(data: Any = None, message: str | None = None, status_code: int = 200):
    """
    Response sukses generik.

    Args:
        data:        Payload utama (dict, list, atau None).
        message:     Pesan opsional.
        status_code: HTTP status code (default 200).

    Returns:
        Flask Response dengan JSON body.
    """
    body: dict = {"success": True}
    if message:
        body["message"] = message
    if data is not None:
        body["data"] = data
    return jsonify(body), status_code


def success_list(
    data: list,
    pagination: dict,
    filters_applied: dict | None = None,
    status_code: int = 200,
):
    """
    Response sukses untuk endpoint list dengan pagination.

    Args:
        data:            List item yang sudah di-serialize.
        pagination:      Dict metadata pagination.
        filters_applied: Dict filter yang sedang aktif (opsional).
        status_code:     HTTP status code (default 200).
    """
    body: dict = {
        "success":    True,
        "data":       data,
        "pagination": pagination,
    }
    if filters_applied is not None:
        body["filters_applied"] = filters_applied
    return jsonify(body), status_code


# ─────────────────────────────────────────────────────────────────────────────
# Error responses
# ─────────────────────────────────────────────────────────────────────────────

def error(message: str, status_code: int = 400, details: Any = None):
    """
    Response error generik.

    Args:
        message:     Pesan error yang human-readable.
        status_code: HTTP status code (default 400).
        details:     Info tambahan opsional (misal: allowed_formats).
    """
    body: dict = {
        "success": False,
        "error":   message,
    }
    if details is not None:
        body["details"] = details
    return jsonify(body), status_code


def not_found(resource: str = "Resource", identifier: Any = None):
    """
    Response 404 Not Found.

    Args:
        resource:   Nama resource (misal: "DetectionHistory").
        identifier: ID atau identifier yang tidak ditemukan.
    """
    msg = f"{resource} tidak ditemukan"
    if identifier is not None:
        msg = f"{resource} dengan id={identifier} tidak ditemukan"
    return error(msg, 404)


def server_error(message: str = "Terjadi kesalahan internal server"):
    """Response 500 Internal Server Error."""
    return error(message, 500)


def validation_error(message: str, details: Any = None):
    """Response 422 Unprocessable Entity untuk validasi input."""
    return error(message, 422, details)


# ─────────────────────────────────────────────────────────────────────────────
# Pagination builder
# ─────────────────────────────────────────────────────────────────────────────

def build_pagination_meta(pagination) -> dict:
    """
    Membangun dict metadata pagination dari objek Flask-SQLAlchemy Pagination.

    Args:
        pagination: Objek hasil query.paginate().

    Returns:
        Dict dengan keys: page, per_page, total, total_pages,
        has_next, has_prev, next_page, prev_page.
    """
    return {
        "page":        pagination.page,
        "per_page":    pagination.per_page,
        "total":       pagination.total,
        "total_pages": pagination.pages,
        "has_next":    pagination.has_next,
        "has_prev":    pagination.has_prev,
        "next_page":   pagination.next_num,
        "prev_page":   pagination.prev_num,
    }
