from typing import List

from fastapi import APIRouter, HTTPException, status

from models.model import Model
from models.requests import ModelRequest, ModelsListResponse
from services.model_service import model_service
from services.lora_service import lora_service

router = APIRouter(
    prefix="/models",
    tags=["models"]
)


@router.get("/", response_model=ModelsListResponse)
async def list_models():
    """List all available models."""
    models = model_service.get_models()
    loras_list = lora_service.get_loras()
    return ModelsListResponse(
        models=[model for model in models if model.details.specialization !=
                "LoRA"] + loras_list,
        active_model=model_service.active_model_id
    )


@router.get("/{model_id}", response_model=Model)
async def get_model(model_id: str):
    """Get information about a specific model."""
    model = model_service.get_model(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with Id {model_id} not found"
        )
    return model


@router.post("/", response_model=Model, status_code=status.HTTP_201_CREATED)
async def add_model(model_request: ModelRequest):
    """Add a new model."""
    try:
        model = model_service.add_model(
            name=model_request.name or model_request.source,
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


@router.put("/active/{model_id}", response_model=Model)
async def set_active_model(model_id: str):
    """Set a model as the active model."""
    if not model_service.set_active_model(model_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with Id {model_id} not found"
        )
    return model_service.get_model(model_id)
