from fastapi import APIRouter
from typing import Optional
from models.openai.create_moderation_request import CreateModerationRequest
from models.openai.create_moderation_response import CreateModerationResponse


router = APIRouter(prefix="/moderations", tags=["Moderations"])


@router.post("/")
async def createModeration(body: CreateModerationRequest) -> CreateModerationResponse:
    """Operation ID: createModeration"""
    raise NotImplementedError("Endpoint not yet implemented")
