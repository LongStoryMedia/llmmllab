"""
Users router for handling user management and authentication.

Note: This router is included in app.py with both non-versioned and versioned paths:
- Non-versioned: /users/...
- Versioned: /v1/users/...
"""

from typing import List
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request

from db import storage
from server.middleware.auth import get_user_id, is_admin

from models.user import User
from models.conversation import Conversation

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/", response_model=List[User])
async def get_users(request: Request):
    """Get all users - admin only"""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not is_admin(request):
        raise HTTPException(status_code=403, detail="Admin access required")

    if not storage.initialized:
        raise HTTPException(status_code=503, detail="Database not initialized")

    if not storage.user_config:
        raise HTTPException(
            status_code=503, detail="User config storage not initialized"
        )

    try:
        users = await storage.user_config.get_all_users()
        # Transform the data to match the User model
        transformed_users = []
        for user in users:
            # Copy the original user data
            user_copy = dict(user)

            # Add the uid field (required by the User model)
            # Using id as the uid since they appear to represent the same data
            user_copy["uid"] = user.get("id", "")

            # Handle NULL username - provide default value or use id
            if user_copy.get("username") is None:
                user_copy["username"] = user_copy.get("id", "unknown_user")

            transformed_users.append(user_copy)

        return transformed_users
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching users: {str(e)}")


@router.get("/{user_id}/conversations", response_model=List[Conversation])
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

    if not storage.initialized:
        raise HTTPException(status_code=503, detail="Database not initialized")

    if not storage.conversation:
        raise HTTPException(
            status_code=503, detail="Conversation storage not initialized"
        )

    return await storage.conversation.get_user_conversations(user_id)
