import os
import logging
import cloudinary
import cloudinary.uploader

logger = logging.getLogger(__name__)

# Configure Cloudinary
CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
API_KEY = os.getenv("CLOUDINARY_API_KEY")
API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

import sys

if CLOUD_NAME and API_KEY and API_SECRET:
    cloudinary.config(
        cloud_name=CLOUD_NAME,
        api_key=API_KEY,
        api_secret=API_SECRET,
        secure=True
    )
    logger.info("Cloudinary configured successfully.")
else:
    print("[Cloudinary Config Warning] Credentials are not completely configured in environment variables.", file=sys.stderr)
    logger.warning("Cloudinary credentials are not completely configured in environment variables.")


def upload_image_to_cloudinary(image_bytes: bytes, filename: str, folder: str = "plantscan") -> str | None:
    """
    Uploads raw image bytes to Cloudinary.

    Args:
        image_bytes: The raw image bytes.
        filename: Original filename to construct public_id or reference.
        folder: The Cloudinary folder to save the image in.

    Returns:
        The secure URL of the uploaded image if successful, otherwise None.
    """
    if not (CLOUD_NAME and API_KEY and API_SECRET):
        print("[Cloudinary Error] Cannot upload: Cloudinary is not configured.", file=sys.stderr)
        logger.error("Cloudinary is not configured. Skipping upload.")
        return None

    try:
        # Construct a simple public ID prefix using original filename base
        base_name = os.path.splitext(filename)[0]
        # Clean special chars from public ID if necessary, Cloudinary uploader can handle raw bytes via file-like objects or byte strings
        response = cloudinary.uploader.upload(
            image_bytes,
            folder=folder,
            resource_type="image",
            filename=filename
        )
        return response.get("secure_url")
    except Exception as e:
        print(f"[Cloudinary Error] Failed to upload image: {e}", file=sys.stderr)
        logger.error(f"Failed to upload image to Cloudinary: {e}")
        return None
