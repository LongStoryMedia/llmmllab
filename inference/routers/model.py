"""
Models router for handling model management and configuration.

"""

from typing import List
from fastapi import APIRouter, HTTPException, Request

from middleware.auth import get_user_id
from config import logger

from models.model import Model
from runner.utils.model_loader import ModelLoader

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/", response_model=List[Model])
async def list_models(request: Request):
    """List all available models."""
    # We're not currently using the user_id for filtering, but we may in the future
    _ = get_user_id(request)

    try:
        # Use ModelLoader for consistent model loading with validation and defaults
        model_loader = ModelLoader()
        models_dict = model_loader.get_available_models()

        # Convert dictionary values to list
        models = list(models_dict.values())

        logger.info(f"Successfully loaded {len(models)} models for API")
        return models

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error loading models: {str(e)}"
        ) from e
