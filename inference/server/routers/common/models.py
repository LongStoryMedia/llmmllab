from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from typing import Optional, Union
from server.middleware.auth import get_user_id
from models.openai import DeleteModelResponse, ListModelsResponse, Model as OpenAIModel
from models.anthropic import (
    ModelListResponse as AnthropicModelListResponse,
    Model as AnthropicModel,
)
from models.model import Model
from models.model_task import ModelTask
from models.model_details import ModelDetails
from models.model_provider import ModelProvider
from utils.logging import llmmllogger


logger = llmmllogger.bind(component="common_models_router")
router = APIRouter(prefix="/models", tags=["Models"])


# Union types for common endpoints
OpenAIModelListResponse = ListModelsResponse

OpenAIModelType = OpenAIModel
AnthropicModelType = AnthropicModel


def to_openai_model(model: Model) -> OpenAIModel:
    """Convert internal Model representation to OpenAI API response format."""
    assert model.id is not None, "Model ID cannot be None"
    return OpenAIModel(
        id=model.id,
        object="model",
        created=int(datetime.now().timestamp()),
        owned_by="llmmllab",
    )


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
async def listModels(
    request: Request,
) -> Union[OpenAIModelListResponse, AnthropicModelListResponse]:
    """Operation ID: listModels (OpenAI) / listModels (Anthropic)"""
    user_id = get_user_id(request)

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in request")

    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{model_id}")
async def getModel(
    model_id: str,
    request: Request,
) -> Union[OpenAIModelType, AnthropicModelType]:
    """Operation ID: getModel (OpenAI) / getModel (Anthropic)"""
    user_id = get_user_id(request)

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in request")

    raise NotImplementedError("Endpoint not yet implemented")


@router.delete("/{model_id}")
async def deleteModel(
    model_id: str,
    request: Request,
) -> DeleteModelResponse:
    """Operation ID: deleteModel (OpenAI)"""
    user_id = get_user_id(request)

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in request")

    raise NotImplementedError("Endpoint not yet implemented")
