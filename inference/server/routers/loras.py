from typing import List

from fastapi import APIRouter, HTTPException, status

from models.model import Model
from models.requests import LoraWeightRequest, ModelRequest, LoraListResponse
from services.lora_service import lora_service
from services.model_service import model_service

router = APIRouter(
    prefix="/loras",
    tags=["loras"]
)


@router.get("/", response_model=LoraListResponse)
async def list_loras():
    """List all available LoRAs."""
    loras = lora_service.get_loras()
    return LoraListResponse(
        loras=loras,
        active_loras=lora_service.active_loras
    )


@router.get("/{lora_id}", response_model=Model)
async def get_lora(lora_id: str):
    """Get information about a specific LoRA."""
    lora = lora_service.get_lora(lora_id)
    if not lora:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"LoRA with Id {lora_id} not found"
        )
    return lora


@router.post("/", response_model=Model, status_code=status.HTTP_201_CREATED)
async def add_lora(request: ModelRequest):
    """Add a new LoRA."""
    try:
        lora = lora_service.add_lora(
            name=request.name or request.source,
            source=request.source,
            description=request.description,
            weight=request.weight
        )
        return lora
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to add LoRA: {str(e)}"
        )


@router.delete("/{lora_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_lora(lora_id: str):
    """Remove a LoRA."""
    if not lora_service.remove_lora(lora_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"LoRA with Id {lora_id} not found or cannot be removed"
        )


@router.put("/{lora_id}/activate", response_model=Model, status_code=status.HTTP_202_ACCEPTED)
async def activate_lora(lora_id: str):
    """Activate a LoRA."""
    lora = lora_service.get_lora(lora_id)
    if not lora:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"LoRA with Id {lora_id} not found"
        )

    if not lora_service.activate_lora(lora_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to activate LoRA {lora_id}"
        )

    # Reload the model to apply LoRA weights
    if model_service.active_pipeline:
        try:
            model_service._apply_active_loras()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to apply LoRA weights: {str(e)}"
            )

    return lora_service.get_lora(lora_id)


@router.put("/{lora_id}/deactivate")
async def deactivate_lora(lora_id: str):
    """Deactivate a LoRA."""
    if not lora_service.deactivate_lora(lora_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"LoRA with Id {lora_id} not found or not active"
        )

    # Reload the model to apply changes
    if model_service.active_pipeline:
        try:
            # For proper deactivation, we need to reload the model
            model_service.load_active_model()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update model after LoRA deactivation: {str(e)}"
            )

    return {"success": True}


@router.put("/{lora_id}/weight", response_model=Model)
async def set_lora_weight(lora_id: str, request: LoraWeightRequest):
    """Set the weight for a LoRA."""
    lora = lora_service.get_lora(lora_id)
    if not lora:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"LoRA with Id {lora_id} not found"
        )

    # Validate weight
    if request.weight < 0 or request.weight > 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Weight must be between 0 and 1"
        )

    if not lora_service.set_lora_weight(lora_id, request.weight):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to set weight for LoRA {lora_id}"
        )

    # Update LoRA weights if active
    if lora_id in lora_service.active_loras and model_service.active_pipeline:
        try:
            model_service._apply_active_loras()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update LoRA weights: {str(e)}"
            )

    return lora_service.get_lora(lora_id)
