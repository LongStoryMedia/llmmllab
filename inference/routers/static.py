"""
Static router for serving static files like images.

"""

import os
import mimetypes
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import IMAGE_DIR

router = APIRouter(prefix="/static", tags=["static"])


def get_media_type(filename: str) -> str:
    """Detect MIME type from file extension."""
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type:
        return mime_type

    # Fallback based on extension
    ext = os.path.splitext(filename)[1].lower()
    fallback_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".svg": "image/svg+xml",
    }

    return fallback_types.get(ext, "application/octet-stream")


@router.get("/images/view/{filename}")
async def serve_image(filename: str):
    """Serve an image file for viewing in browser"""
    file_path = os.path.join(IMAGE_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")

    # Detect proper MIME type
    media_type = get_media_type(filename)

    return FileResponse(
        file_path,
        media_type=media_type,
        filename=filename,
    )


@router.get("/images/download/{filename}")
async def download_image(filename: str):
    """Download an image file"""
    file_path = os.path.join(IMAGE_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
