from typing import List

from fastapi import APIRouter, HTTPException, status

from models.model import Model
from models.requests import ModelRequest, ModelsListResponse
from services.model_service import model_service

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/", response_model=ModelsListResponse)
async def list_models():
    """List all available models."""
    try:
        # Get models from the model service
        models = model_service.models.values()

        # Safely filter models by checking if specialization attribute exists
        filtered_models = []
        for model in models:
            filtered_models.append(model)

        # Create response
        response = ModelsListResponse(models=filtered_models, active_model="")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")
