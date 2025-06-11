import base64
import os
import uuid
from io import BytesIO

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import numpy as np
import torch
from PIL import Image

from config import IMAGE_DIR
from models.requests import PromptRequest
from services.image_generator import generator
from services.hardware_manager import hardware_manager

router = APIRouter()


@router.post("/generate-image")
async def generate_image(request: PromptRequest):
    """Generate an image from a prompt."""
    try:
        print(
            f"Processing image generation request with prompt: '{request.prompt[:30]}...'")

        # Use hardware manager to check resources
        print(hardware_manager.get_memory_status_str())

        # If memory is low, enable low memory mode
        if hardware_manager.is_low_memory():
            print("Low GPU memory detected, forcing low memory mode")
            request.low_memory_mode = True

        # Use custom parameters if specified
        params = {}
        if request.width:
            params["width"] = request.width
        if request.height:
            params["height"] = request.height
        if request.inference_steps:
            params["num_inference_steps"] = request.inference_steps
        if request.guidance_scale:
            params["guidance_scale"] = request.guidance_scale

        # Validate prompt
        if not request.prompt or len(request.prompt.strip()) == 0:
            raise ValueError("Prompt cannot be empty")

        # Generate image using the generator service
        image = generator.generate(request.prompt, **params)

        if not image:
            raise ValueError("Model returned empty image")

        # Convert to PIL Image if necessary
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if image.ndim == 3 and image.shape[2] in [3, 4]:
                image = Image.fromarray(
                    (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8))
            else:
                raise ValueError("Unexpected numpy array format")
        elif torch.is_tensor(image):
            # Convert tensor to PIL Image
            if image.ndim == 3:
                if image.shape[0] in [1, 3, 4]:  # [C, H, W]
                    image = image.permute(1, 2, 0)  # Convert to [H, W, C]
                image_np = image.detach().cpu().numpy()
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                image = Image.fromarray(image_np.astype(np.uint8))
            else:
                raise ValueError("Unexpected tensor format")
        elif not hasattr(image, 'save'):
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Handle case where image might be a list
        if isinstance(image, list):
            if not image:
                raise ValueError("Model returned empty image list")
            image = image[0]  # Take the first image from the list

        # Save the image to disk
        filename = uuid.uuid4().hex + ".png"
        file_path = os.path.join(IMAGE_DIR, filename)
        image.save(file_path)

        # Convert PIL image to base64 string
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_data = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

        # Construct download URL
        download_path = f"/download/{filename}"

        return {
            "image": img_data,
            "download": download_path
        }
    except Exception as e:
        import traceback
        print(f"Error in generate_image: {type(e).__name__}: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Image generation failed: {str(e)}"
        )


@router.get("/download/{filename}")
async def download_image(filename: str):
    """Download a generated image by filename."""
    file_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="image/png")
