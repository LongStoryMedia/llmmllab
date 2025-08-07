from typing import List, Optional
from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
import time
import uuid

from server.auth import get_user_id, is_admin
from server.routers.chat import ConversationResponse, ListConversationsResponse

router = APIRouter(prefix="/users", tags=["users"])


class User(BaseModel):
    id: str
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    is_admin: bool = False
    created_at: float
    updated_at: float


class UsersResponse(BaseModel):
    users: List[User]


@router.get("/", response_model=UsersResponse)
async def get_users(request: Request):
    """Get all users - admin only"""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not is_admin(request):
        raise HTTPException(status_code=403, detail="Admin access required")

    # Mock implementation - in a real scenario, fetch from database
    users = [
        User(
            id="1",
            username="admin",
            email="admin@example.com",
            full_name="Admin User",
            is_admin=True,
            created_at=time.time() - 3600 * 24,
            updated_at=time.time() - 1800,
        ),
        User(
            id="2",
            username="user1",
            email="user1@example.com",
            full_name="Regular User",
            is_admin=False,
            created_at=time.time() - 3600 * 12,
            updated_at=time.time() - 900,
        ),
    ]

    return UsersResponse(users=users)


@router.get("/{user_id}/conversations", response_model=ListConversationsResponse)
async def get_conversations_for_user(user_id: str, request: Request):
    """Get all conversations for a specific user - admin only"""
    caller_id = get_user_id(request)
    if not caller_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Users can view their own conversations, admins can view any user's conversations
    if caller_id != user_id and not is_admin(request):
        raise HTTPException(
            status_code=403,
            detail="Admin access required to view other users' conversations",
        )

    # Mock implementation - in a real scenario, fetch from database
    conversations = [
        ConversationResponse(
            id=str(uuid.uuid4()),
            user_id=user_id,
            title="Sample Conversation 1",
            created_at=time.time() - 3600,
            updated_at=time.time() - 1800,
            model_id="gpt-4",
            message_count=5,
            is_pinned=False,
        ),
        ConversationResponse(
            id=str(uuid.uuid4()),
            user_id=user_id,
            title="Sample Conversation 2",
            created_at=time.time() - 7200,
            updated_at=time.time() - 3600,
            model_id="gpt-3.5-turbo",
            message_count=12,
            is_pinned=True,
        ),
    ]

    return ListConversationsResponse(conversations=conversations)
