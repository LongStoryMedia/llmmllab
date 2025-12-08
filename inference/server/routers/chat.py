"""
Simplified Chat router that delegates to composer interface.
All chat logic has been moved to the composer module for clean architectural separation.

Note: This router is included in app.py with both non-versioned and versioned paths:
- Non-versioned: /chat/...
- Versioned: /v1/chat/...
"""

import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from server.middleware.auth import get_request_id, get_user_id, is_admin
from db import storage  # Import database storage
from models import (
    MessageRole,
    ChatResponse,
    Message,
)
from utils import extract_text_from_message  # Import logging utility
from utils.logging import llmmllogger, serialize_event_data
from utils.message_transformation import transform_file_content_to_documents

# Import composer interface and streaming state management
import composer


logger = llmmllogger.bind(component="chat_router")
router = APIRouter(prefix="/chat", tags=["chat"])


async def composer_chat_completion(
    user_id: str, conversation_id: int, request_id: str
) -> AsyncIterator[str]:
    """Handle chat completions by delegating to composer interface."""
    # Compose workflow for user
    workflow = await composer.compose_workflow(user_id)

    # Create initial state (conversation_id is already validated)
    initial_state = await composer.create_initial_state(user_id, conversation_id)

    logger.info(f"Starting workflow execution for request {request_id}")

    async for event in composer.execute_workflow(initial_state, workflow):
        print(
            extract_text_from_message(event.message) if event.message else "",
            flush=True,
            end="",
        )  # Debug print
        yield f"{event.model_dump_json()}"


@router.post("/completions", response_model=ChatResponse)
async def chat_completion(
    msg: Message,
    request: Request,
):
    """
    Handle chat completions with composer integration.
    Uses composer workflow orchestration for enhanced AI capabilities.
    """
    # Early validation and setup
    user_id = get_user_id(request)
    request_id = get_request_id(request)

    # Validate inputs early
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    if not msg.conversation_id:
        raise HTTPException(status_code=400, detail="Conversation ID not found")
    if not msg or msg.role != MessageRole.USER:
        raise HTTPException(status_code=400, detail="Invalid user message")
    if not request_id:
        raise HTTPException(status_code=400, detail="Request ID not found")

    logger.info(f"Processing chat completion request {request_id} for user {user_id}")

    try:
        # Transform file content to documents before storing
        msg = await transform_file_content_to_documents(msg, user_id)

        await storage.get_service(storage.message).add_message(msg)
        return StreamingResponse(
            composer_chat_completion(user_id, msg.conversation_id, request_id),  # type: ignore
            media_type="application/json",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        logger.error(f"Error in composer chat completion: {e}", exc_info=True)

        # Provide specific error messages
        error_detail = f"Error in chat completion: {str(e)}"
        if "composer service not initialized" in str(e).lower():
            error_detail = "AI service not ready. Please try again in a moment."
        elif "workflow construction" in str(e).lower():
            error_detail = (
                "Unable to create AI workflow. Please check your configuration."
            )
        elif "unknown model architecture" in str(e):
            error_detail = (
                "Model architecture not supported. Please try a different model."
            )
        elif "Failed to create llama_context" in str(e):
            error_detail = (
                "Model failed to load. This may be due to insufficient memory."
            )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail,
        ) from e


@router.get("/admin")
async def admin_only(request: Request):
    """
    Admin-only endpoint to demonstrate role-based access control.
    Only users with admin privileges can access this endpoint.
    """
    # Check if user is admin
    if not is_admin(request):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required for this endpoint",
        )

    user_id = get_user_id(request)
    request_id = get_request_id(request)

    logger.info(f"Admin access granted for user {user_id}, request {request_id}")

    return {
        "status": "success",
        "message": "Admin access granted",
        "user_id": user_id,
        "request_id": request_id,
    }
