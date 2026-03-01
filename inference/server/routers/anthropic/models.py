from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from typing import Optional
from server.middleware.auth import get_user_id
from models.anthropic import Model as AnthropicModel, ModelListResponse
from models.model import Model
from utils.logging import llmmllogger


logger = llmmllogger.bind(component="anthropic_models_router")
router = APIRouter(prefix="/models", tags=["Models"])


def to_anthropic_model(model: Model) -> AnthropicModel:
    """Convert internal Model representation to Anthropic API response format."""
    assert model.id is not None, "Model ID cannot be None"
    return AnthropicModel(
        id=model.id,
        type="model",
        display_name=model.name if hasattr(model, "name") and model.name else model.id,
        created_at=datetime.now(),
    )


@router.get("/")
async def listModels(request: Request) -> ModelListResponse:
    """Operation ID: listModels"""
    _ = get_user_id(request)

    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{model_id}")
async def getModel(model_id: str, request: Request) -> AnthropicModel:
    """Operation ID: getModel"""
    _ = get_user_id(request)

    raise NotImplementedError("Endpoint not yet implemented")
