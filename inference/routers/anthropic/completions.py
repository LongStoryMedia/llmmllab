from fastapi import APIRouter, HTTPException, Request
from typing import Union
from fastapi.responses import StreamingResponse

from middleware.auth import get_user_id
from models.anthropic.create_completion_request import CreateCompletionRequest
from models.anthropic.completion_response import CompletionResponse
from utils.logging import llmmllogger


logger = llmmllogger.bind(component="anthropic_completions_router")
router = APIRouter(prefix="/complete", tags=["Completions"])


@router.post("/")
async def createCompletion(
    request: Request,
    body: CreateCompletionRequest,
):
    """Operation ID: createCompletion"""
    user_id = get_user_id(request)

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in request")

    raise NotImplementedError("Endpoint not yet implemented")
