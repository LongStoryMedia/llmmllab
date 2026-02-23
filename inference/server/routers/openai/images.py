from fastapi import APIRouter
from typing import Optional
from models.openai.create_image_request import CreateImageRequest
from models.openai.images_response import ImagesResponse


router = APIRouter(prefix="/images", tags=["Images"])


@router.post("/edits")
async def createImageEdit() -> ImagesResponse:
    """Operation ID: createImageEdit"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/generations")
async def createImage(body: CreateImageRequest) -> ImagesResponse:
    """Operation ID: createImage"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/variations")
async def createImageVariation() -> ImagesResponse:
    """Operation ID: createImageVariation"""
    raise NotImplementedError("Endpoint not yet implemented")
