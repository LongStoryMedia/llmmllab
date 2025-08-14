"""
Image router for handling image generation, editing, and manipulation.

Note: This router is included in app.py with both non-versioned and versioned paths:
- Non-versioned: /images/...
- Versioned: /v1/images/...
"""

import base64
import os
import time
from pydoc import text
import uuid
from io import BytesIO
from typing import List, Optional, Dict, Any

from fastapi import (
    APIRouter,
    HTTPException,
    BackgroundTasks,
    UploadFile,
    Request,
    status,
)
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

from server.config import IMAGE_DIR, IMAGE_RETENTION_HOURS, logger
from models.requests import PromptRequest
from server.services.image_generator import image_generator
from server.auth import get_user_id, can_access, is_admin


# Define models for image-related operations
class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field(
        "", description="Negative prompt for image generation"
    )
    width: Optional[int] = Field(512, description="Width of the generated image")
    height: Optional[int] = Field(512, description="Height of the generated image")
    num_inference_steps: Optional[int] = Field(
        50, description="Number of inference steps"
    )
    guidance_scale: Optional[float] = Field(7.5, description="Guidance scale")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    model: Optional[str] = Field(None, description="Model to use for generation")


class ImageEditRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image editing")
    negative_prompt: Optional[str] = Field(
        "", description="Negative prompt for image editing"
    )
    image_id: str = Field(..., description="ID of the image to edit")
    mask_id: Optional[str] = Field(None, description="ID of the mask image")
    strength: Optional[float] = Field(0.75, description="Strength of the edit")
    num_inference_steps: Optional[int] = Field(
        50, description="Number of inference steps"
    )
    guidance_scale: Optional[float] = Field(7.5, description="Guidance scale")


class ImageMetadata(BaseModel):
    id: str
    user_id: str
    filename: str
    created_at: float
    width: int
    height: int
    prompt: Optional[str] = None
    model: Optional[str] = None
    view_url: Optional[str] = None
    download_url: Optional[str] = None


class ImageListResponse(BaseModel):
    images: List[ImageMetadata]
    count: int


router = APIRouter(prefix="/images", tags=["images"])


os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


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
            image.save(img_byte_arr, format="PNG")
            img_data = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

            return {
                "status": "complete",
                "image": img_data,
                "download": f"/download/{request_id}.png",
            }
        else:
            # Image is still processing
            return {
                "status": "processing",
                "message": "Your image is still being generated.",
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/download/{filename}")
async def download_image(filename: str):
    """Download a generated image by filename."""
    file_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="image/png")


@router.get("/download/user/{user_id}/{filename}")
async def download_user_image(request: Request, user_id: str, filename: str):
    """Download a user-specific image by filename with access control."""
    # Check if the requesting user has access to the target user's resources
    if not can_access(request, user_id):
        raise HTTPException(
            status_code=403,
            detail="Access denied: You don't have permission to access this image",
        )

    file_path = os.path.join(IMAGE_DIR, "originals", user_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Determine content type based on file extension
    content_type = "image/png"
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
        content_type = "image/jpeg"

    return FileResponse(file_path, media_type=content_type)


@router.post("/store-image")
async def store_image(image: UploadFile, request: Request):
    """Store a generated image."""
    try:
        # Get user ID from request
        user_id = get_user_id(request)
        if not user_id:
            raise HTTPException(status_code=401, detail="Authentication required")

        # ensure the image directory exists
        user_image_dir = os.path.join(IMAGE_DIR, "originals", user_id)
        os.makedirs(user_image_dir, exist_ok=True)

        # Save the uploaded image
        file_path = os.path.join(
            user_image_dir,
            image.filename if image.filename else f"{uuid.uuid4()}.png",
        )
        with open(file_path, "wb") as f:
            f.write(await image.read())
        return {"status": "success", "file_path": file_path, "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{target_user_id}", response_model=ImageListResponse)
async def get_user_images(request: Request, target_user_id: str):
    """
    Get images for a specific user.
    Only the user themselves or admins can access this endpoint.
    """
    # Check if the requesting user has access to the target user's resources
    if not can_access(request, target_user_id):
        raise HTTPException(
            status_code=403,
            detail="Access denied: You don't have permission to view these images",
        )

    try:
        user_image_dir = os.path.join(IMAGE_DIR, "originals", target_user_id)
        if not os.path.exists(user_image_dir):
            return ImageListResponse(images=[], count=0)

        # Get list of images for the user
        image_files = [
            f
            for f in os.listdir(user_image_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]
        image_paths = [os.path.join(user_image_dir, f) for f in image_files]

        # Create image metadata list
        images = []
        for path in image_paths:
            filename = os.path.basename(path)
            image_id = filename.split(".")[0]  # Assume filename is UUID.extension

            # Get image dimensions
            try:
                with Image.open(path) as img:
                    width, height = img.size
            except:
                width, height = 0, 0

            images.append(
                ImageMetadata(
                    id=image_id,
                    user_id=target_user_id,
                    filename=filename,
                    created_at=os.path.getctime(path),
                    width=width,
                    height=height,
                    prompt="",  # In a real implementation, retrieve this from metadata storage
                    model="",  # In a real implementation, retrieve this from metadata storage
                    view_url=f"/images/view/{target_user_id}/{filename}",
                    download_url=f"/images/download/user/{target_user_id}/{filename}",
                )
            )

        return ImageListResponse(images=images, count=len(images))
    except Exception as e:
        logger.error(f"Error fetching user images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching images: {str(e)}")


@router.post("/generate", response_model=ImageMetadata)
async def generate_image(
    request: ImageGenerationRequest, req: Request, background_tasks: BackgroundTasks
):
    """Generate an image based on the provided prompt."""
    user_id = get_user_id(req)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        # In a real implementation, call the image generator service
        # For now, create a placeholder response
        image_id = str(uuid.uuid4())
        filename = f"{image_id}.png"

        # Mock image generation - in a real implementation, use Stable Diffusion or similar
        logger.info(
            f"Generating image for user {user_id} with prompt: {request.prompt}"
        )

        # Return metadata immediately, actual generation would happen in background
        return ImageMetadata(
            id=image_id,
            user_id=user_id,
            filename=filename,
            created_at=time.time(),
            width=request.width or 512,  # Use default if None
            height=request.height or 512,  # Use default if None
            prompt=request.prompt,
            model=request.model or "default-model",
            view_url=f"/images/view/{user_id}/{filename}",
            download_url=f"/images/download/user/{user_id}/{filename}",
        )
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Image generation failed: {str(e)}"
        )


@router.post("/edit", response_model=ImageMetadata)
async def edit_image(
    request: ImageEditRequest, req: Request, background_tasks: BackgroundTasks
):
    """Edit an existing image based on the provided prompt and mask."""
    user_id = get_user_id(req)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        # In a real implementation, verify the image belongs to the user
        # For now, create a placeholder response
        image_id = str(uuid.uuid4())
        filename = f"{image_id}.png"

        # Mock image editing - in a real implementation, use Stable Diffusion inpainting or similar
        logger.info(
            f"Editing image {request.image_id} for user {user_id} with prompt: {request.prompt}"
        )

        # Return metadata immediately, actual editing would happen in background
        return ImageMetadata(
            id=image_id,
            user_id=user_id,
            filename=filename,
            created_at=time.time(),
            width=512,  # Placeholder values
            height=512,  # Placeholder values
            prompt=request.prompt,
            model="default-model",
            view_url=f"/images/view/{user_id}/{filename}",
            download_url=f"/images/download/user/{user_id}/{filename}",
        )
    except Exception as e:
        logger.error(f"Error editing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image editing failed: {str(e)}")


@router.delete("/{image_id}")
async def delete_image(image_id: str, request: Request):
    """Delete an image by ID."""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        # In a real implementation, find the image path from the database or file storage
        user_image_dir = os.path.join(IMAGE_DIR, "originals", user_id)

        # Check common image extensions
        for ext in [".png", ".jpg", ".jpeg"]:
            image_path = os.path.join(user_image_dir, f"{image_id}{ext}")
            if os.path.exists(image_path):
                # Verify ownership (only the owner or admin can delete)
                if not can_access(request, user_id):
                    raise HTTPException(
                        status_code=403,
                        detail="Access denied: You don't have permission to delete this image",
                    )

                # Delete the file
                os.remove(image_path)
                logger.info(f"Deleted image {image_id} for user {user_id}")
                return {"status": "success", "message": f"Image {image_id} deleted"}

        # If we get here, the image wasn't found
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found")
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error deleting image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete image: {str(e)}")


# Update existing endpoints to match the new router prefix
@router.get("/status/{request_id}")
async def check_generation_status(request_id: str):
    """Check the status of an image generation request."""
    try:
        # Check if the image exists
        file_path = os.path.join(IMAGE_DIR, f"{request_id}.png")
        if os.path.exists(file_path):
            # Image is ready, return download info
            # Convert PIL image to base64 string
            image = Image.open(file_path)
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format="PNG")
            img_data = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

            return {
                "status": "complete",
                "image": img_data,
                "download": f"/images/download/{request_id}.png",
            }
        else:
            # Image is still processing
            return {
                "status": "processing",
                "message": "Your image is still being generated.",
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}
