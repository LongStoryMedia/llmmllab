from fastapi import APIRouter
from typing import Optional
from models.openai.conversation_item import ConversationItem
from models.openai.conversation_item_list import ConversationItemList
from models.openai.conversation_resource import ConversationResource
from models.openai.create_conversation_body import CreateConversationBody
from models.openai.deleted_conversation_resource import DeletedConversationResource
from models.openai.update_conversation_body import UpdateConversationBody


router = APIRouter(prefix="/conversations", tags=["Conversations"])


@router.post("/")
async def createConversation(body: CreateConversationBody) -> ConversationResource:
    """Operation ID: createConversation"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.delete("/{conversation_id}")
async def deleteConversation(conversation_id: str) -> DeletedConversationResource:
    """Operation ID: deleteConversation"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{conversation_id}")
async def getConversation(conversation_id: str) -> ConversationResource:
    """Operation ID: getConversation"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{conversation_id}")
async def updateConversation(
    conversation_id: str, body: UpdateConversationBody
) -> ConversationResource:
    """Operation ID: updateConversation"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{conversation_id}/items")
async def listConversationItems(conversation_id: str) -> ConversationItemList:
    """Operation ID: listConversationItems"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{conversation_id}/items")
async def createConversationItems(conversation_id: str) -> ConversationItemList:
    """Operation ID: createConversationItems"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.delete("/{conversation_id}/items/{item_id}")
async def deleteConversationItem(
    conversation_id: str, item_id: str
) -> ConversationResource:
    """Operation ID: deleteConversationItem"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{conversation_id}/items/{item_id}")
async def getConversationItem(conversation_id: str, item_id: str) -> ConversationItem:
    """Operation ID: getConversationItem"""
    raise NotImplementedError("Endpoint not yet implemented")
