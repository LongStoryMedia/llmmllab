from fastapi import APIRouter
from typing import Optional
from models.openai.chat_session_resource import ChatSessionResource
from models.openai.create_chat_session_body import CreateChatSessionBody
from models.openai.deleted_thread_resource import DeletedThreadResource
from models.openai.thread_item_list_resource import ThreadItemListResource
from models.openai.thread_list_resource import ThreadListResource
from models.openai.thread_resource import ThreadResource


router = APIRouter(prefix="/chatkit", tags=["Chatkit"])


@router.post("/sessions")
async def CreateChatSessionMethod(body: CreateChatSessionBody) -> ChatSessionResource:
    """Operation ID: CreateChatSessionMethod"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/sessions/{session_id}/cancel")
async def CancelChatSessionMethod(session_id: str) -> ChatSessionResource:
    """Operation ID: CancelChatSessionMethod"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/threads")
async def ListThreadsMethod() -> ThreadListResource:
    """Operation ID: ListThreadsMethod"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.delete("/threads/{thread_id}")
async def DeleteThreadMethod(thread_id: str) -> DeletedThreadResource:
    """Operation ID: DeleteThreadMethod"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/threads/{thread_id}")
async def GetThreadMethod(thread_id: str) -> ThreadResource:
    """Operation ID: GetThreadMethod"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/threads/{thread_id}/items")
async def ListThreadItemsMethod(thread_id: str) -> ThreadItemListResource:
    """Operation ID: ListThreadItemsMethod"""
    raise NotImplementedError("Endpoint not yet implemented")
