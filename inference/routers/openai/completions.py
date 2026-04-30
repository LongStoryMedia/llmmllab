from fastapi import APIRouter
from typing import Optional
from models.openai.create_completion_request import CreateCompletionRequest
from models.openai.create_completion_response import CreateCompletionResponse


router = APIRouter(prefix="/completions", tags=["Completions"])


@router.post("/")
async def createCompletion(body: CreateCompletionRequest) -> CreateCompletionResponse:
    """Operation ID: createCompletion"""
    raise NotImplementedError("Endpoint not yet implemented")
