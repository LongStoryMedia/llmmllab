import os
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse
from typing import Optional

from server.config import IMAGE_DIR

router = APIRouter(prefix="/static", tags=["static"])


@router.get("/images/view/{filename}")
async def serve_image(filename: str):
    """Serve an image file for viewing in browser"""
    file_path = os.path.join(IMAGE_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(
        file_path,
        media_type="image/png",  # Adjust based on your image types
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
