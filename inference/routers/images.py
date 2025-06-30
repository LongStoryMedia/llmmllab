import base64
import os
from pydoc import text
import uuid
from io import BytesIO

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from PIL import Image

from config import IMAGE_DIR
from models.requests import PromptRequest
from services.image_generator import image_generator

router = APIRouter()


os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


@router.post("/generate-image")
async def generate_image(request: PromptRequest, background_tasks: BackgroundTasks):
    """Generate an image from a prompt."""
    try:
        print(f"Processing image generation request with prompt: '{request.prompt[:30]}...'")

        # Initialize response data for async operation
        response_data = {
            "status": "processing",
            "message": "Your image is being generated. Please check back with the provided request_id."
        }

        # Generate unique request ID
        request_id = uuid.uuid4().hex
        response_data["request_id"] = request_id
        # Schedule the image generation in the background
        background_tasks.add_task(
            image_generator.generate_async,
            request.prompt,
            image_generator.process_and_save_generated_image,
            width=request.width or 1024,
            height=request.height or 1024,
            num_inference_steps=request.inference_steps or 20,
            guidance_scale=request.guidance_scale or 7.0,
            negative_prompt=request.negative_prompt,
            request_id=request_id,
        )

        # Return an immediate response
        return response_data

    except Exception as e:
        import traceback
        print(f"Error in generate_image: {type(e).__name__}: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Image generation failed: {str(e)}"
        )


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
