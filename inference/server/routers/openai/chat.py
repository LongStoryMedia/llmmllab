import json

from fastapi import APIRouter, Request
from typing import Optional
from models.openai.chat_completion_deleted import ChatCompletionDeleted
from models.openai.chat_completion_list import ChatCompletionList
from models.openai.chat_completion_message_list import ChatCompletionMessageList
from models.openai.create_chat_completion_request import CreateChatCompletionRequest
from models.openai.create_chat_completion_response import CreateChatCompletionResponse
from utils.logging import llmmllogger


logger = llmmllogger.bind(component="openai_chat_router")
router = APIRouter(prefix="/chat", tags=["Chat"])


@router.get("/completions")
async def listChatCompletions() -> ChatCompletionList:
    """Operation ID: listChatCompletions"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/completions")
async def createChatCompletion(
    body: CreateChatCompletionRequest,
) -> CreateChatCompletionResponse:
    """Operation ID: createChatCompletion"""
    logger.info(json.dumps(body.model_dump(), indent=2))
    raise NotImplementedError("Endpoint not yet implemented")


@router.delete("/completions/{completion_id}")
async def deleteChatCompletion(completion_id: str) -> ChatCompletionDeleted:
    """Operation ID: deleteChatCompletion"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/completions/{completion_id}")
async def getChatCompletion(completion_id: str) -> CreateChatCompletionResponse:
    """Operation ID: getChatCompletion"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/completions/{completion_id}")
async def updateChatCompletion(completion_id: str) -> CreateChatCompletionResponse:
    """Operation ID: updateChatCompletion"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/completions/{completion_id}/messages")
async def getChatCompletionMessages(completion_id: str) -> ChatCompletionMessageList:
    """Operation ID: getChatCompletionMessages"""
    raise NotImplementedError("Endpoint not yet implemented")
