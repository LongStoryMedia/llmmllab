from fastapi import APIRouter
from typing import Optional
from models.openai.assistant_object import AssistantObject
from models.openai.create_assistant_request import CreateAssistantRequest
from models.openai.delete_assistant_response import DeleteAssistantResponse
from models.openai.list_assistants_response import ListAssistantsResponse
from models.openai.modify_assistant_request import ModifyAssistantRequest


router = APIRouter(prefix="/assistants", tags=["Assistants"])


@router.get("/")
async def listAssistants() -> ListAssistantsResponse:
    """Operation ID: listAssistants"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/")
async def createAssistant(body: CreateAssistantRequest) -> AssistantObject:
    """Operation ID: createAssistant"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.delete("/{assistant_id}")
async def deleteAssistant(assistant_id: str) -> DeleteAssistantResponse:
    """Operation ID: deleteAssistant"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{assistant_id}")
async def getAssistant(assistant_id: str) -> AssistantObject:
    """Operation ID: getAssistant"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{assistant_id}")
async def modifyAssistant(
    assistant_id: str, body: ModifyAssistantRequest
) -> AssistantObject:
    """Operation ID: modifyAssistant"""
    raise NotImplementedError("Endpoint not yet implemented")
