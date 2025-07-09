import base64
import os
from pydoc import text
import uuid
from io import BytesIO

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from config import IMAGE_DIR
from models.requests import PromptRequest
from services.image_generator import image_generator

router = APIRouter()


os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


@router.get("/check-image-status/{request_id}")
async def check_image_status(request_id: str):
    """Check the status of an image generation request."""
    try:
        # Check if the image exists
        file_path = os.path.join(IMAGE_DIR, f"{request_id}.png")
        if os.path.exists(file_path):
            # Image is ready, return download info
            # Convert PIL image to base64 string
            image = Image.open(file_path)
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_data = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

            return {
                "status": "complete",
                "image": img_data,
                "download": f"/download/{request_id}.png"
            }
        else:
            # Image is still processing
            return {
                "status": "processing",
                "message": "Your image is still being generated."
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@router.get("/download/{filename}")
async def download_image(filename: str):
    """Download a generated image by filename."""
    file_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="image/png")


@router.post("/store-image")
async def store_image(image: UploadFile):
    """Store a generated image."""
    try:
        # ensure the image directory exists
        os.makedirs(os.path.join(IMAGE_DIR, "originals"), exist_ok=True)
        # Save the uploaded image
        file_path = os.path.join(IMAGE_DIR, "originals", image.filename if image.filename else f"{uuid.uuid4()}.png")
        with open(file_path, "wb") as f:
            f.write(await image.read())
        return {"status": "success", "file_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
