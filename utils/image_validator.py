"""
image_validator.py
──────────────────
Validasi file gambar yang di-upload ke endpoint /api/predict.

Melakukan dua lapis validasi:
  1. Ekstensi file (cepat, sebelum membaca bytes)
  2. Magic bytes / file header (akurat, setelah membaca bytes)
     — mencegah file berbahaya yang di-rename menjadi .jpg/.png
"""

from werkzeug.datastructures import FileStorage

# ─────────────────────────────────────────────────────────────────────────────
# Konstanta
# ─────────────────────────────────────────────────────────────────────────────

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_FILE_SIZE_MB    = MAX_FILE_SIZE_BYTES // (1024 * 1024)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

# Magic bytes untuk setiap format gambar yang didukung
# Format: {extension: [(offset, bytes_to_match), ...]}
_MAGIC_SIGNATURES: dict[str, list[tuple[int, bytes]]] = {
    'jpg':  [(0, b'\xff\xd8\xff')],
    'jpeg': [(0, b'\xff\xd8\xff')],
    'png':  [(0, b'\x89PNG\r\n\x1a\n')],
    'webp': [(0, b'RIFF'), (8, b'WEBP')],
    'bmp':  [(0, b'BM')],
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class ImageValidationError(ValueError):
    """Raised ketika validasi gambar gagal."""
    pass


def validate_image_file(file: FileStorage) -> bytes:
    """
    Validasi lengkap file gambar dan kembalikan bytes-nya.

    Langkah validasi:
      1. Pastikan field 'file' ada dan tidak kosong.
      2. Validasi ekstensi file.
      3. Validasi ukuran file (maks 10 MB).
      4. Validasi magic bytes (file header).

    Args:
        file: Objek FileStorage dari request.files['file'].

    Returns:
        bytes: Isi file gambar yang sudah tervalidasi.

    Raises:
        ImageValidationError: Jika salah satu validasi gagal.
    """
    # ── 1. Nama file tidak boleh kosong ───────────────────────────────────────
    if not file or file.filename == '':
        raise ImageValidationError("Tidak ada file yang dipilih")

    # ── 2. Validasi ekstensi ──────────────────────────────────────────────────
    ext = _get_extension(file.filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            f"Format file '{ext or 'tidak diketahui'}' tidak didukung. "
            f"Format yang diizinkan: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # ── 3. Validasi ukuran ────────────────────────────────────────────────────
    file.seek(0, 2)                    # seek ke akhir
    file_size = file.tell()
    file.seek(0)                       # reset ke awal

    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = round(file_size / (1024 * 1024), 2)
        raise ImageValidationError(
            f"Ukuran file ({size_mb} MB) melebihi batas maksimum {MAX_FILE_SIZE_MB} MB"
        )

    if file_size == 0:
        raise ImageValidationError("File kosong (0 bytes)")

    # ── 4. Baca bytes & validasi magic bytes ──────────────────────────────────
    try:
        image_bytes = file.read()
    except Exception as exc:
        raise ImageValidationError(f"Gagal membaca file: {exc}") from exc

    if not _validate_magic_bytes(image_bytes, ext):
        raise ImageValidationError(
            f"Konten file tidak sesuai dengan ekstensi .{ext}. "
            "Pastikan file adalah gambar yang valid."
        )

    return image_bytes


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_extension(filename: str) -> str:
    """Ekstrak ekstensi file dalam lowercase tanpa titik."""
    if '.' not in filename:
        return ''
    return filename.rsplit('.', 1)[-1].lower()


def _validate_magic_bytes(data: bytes, ext: str) -> bool:
    """
    Periksa apakah bytes awal file cocok dengan magic signature format-nya.

    Args:
        data: Bytes file.
        ext:  Ekstensi file (lowercase, tanpa titik).

    Returns:
        True jika valid, False jika tidak cocok.
    """
    signatures = _MAGIC_SIGNATURES.get(ext)
    if not signatures:
        # Ekstensi tidak ada di daftar magic — loloskan (sudah dicek di step 2)
        return True

    for offset, magic in signatures:
        end = offset + len(magic)
        if len(data) < end:
            return False
        if data[offset:end] != magic:
            return False

    return True
