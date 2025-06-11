from typing import List

from fastapi import APIRouter, HTTPException, status

from models.models import ModelInfo, ModelAddRequest, ModelListResponse
from services.model_service import model_service

router = APIRouter(
    prefix="/models",
    tags=["models"]
)


@router.get("/", response_model=ModelListResponse)
async def list_models():
    """List all available models."""
    models = model_service.get_models()
    return {
        "models": models,
        "active_model": model_service.active_model_id
    }


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """Get information about a specific model."""
    model = model_service.get_model(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with Id {model_id} not found"
        )
    return model


@router.post("/", response_model=ModelInfo, status_code=status.HTTP_201_CREATED)
async def add_model(model_request: ModelAddRequest):
    """Add a new model."""
    try:
        model = model_service.add_model(
            name=model_request.name,
            source=model_request.source,
            description=model_request.description
        )
        return model
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to add model: {str(e)}"
        )


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_model(model_id: str):
    """Remove a model."""
    if not model_service.remove_model(model_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with Id {model_id} not found or cannot be removed"
        )


@router.put("/active/{model_id}", response_model=ModelInfo)
async def set_active_model(model_id: str):
    """Set a model as the active model."""
    if not model_service.set_active_model(model_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with Id {model_id} not found"
        )
    return model_service.get_model(model_id)
