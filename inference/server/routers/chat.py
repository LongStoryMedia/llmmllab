"""
Simplified Chat router that delegates to composer interface.
All chat logic has been moved to the composer module for clean architectural separation.

Note: This router is included in app.py with both non-versioned and versioned paths:
- Non-versioned: /chat/...
- Versioned: /v1/chat/...
"""

import json
from typing import AsyncGenerator

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from models import ChatResponse, Message, MessageRole
from server.middleware.auth import get_request_id, get_user_id, is_admin
from server.config import logger  # Import logger from config
from db import storage  # Import database storage

# Import composer interface
import composer

router = APIRouter(prefix="/chat", tags=["chat"])


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

    logger.info(f"Processing chat completion request {request_id} for user {user_id}")

    try:
        # Store the user message in database first
        if storage.message:
            await storage.message.add_message(msg)

        # Direct composer workflow orchestration
        async def composer_chat_completion() -> AsyncGenerator[str, None]:
            """Handle chat completions by delegating to composer interface."""
            try:
                # Initialize composer service if needed
                await composer.initialize_composer()

                # Compose workflow for user
                workflow = await composer.compose_workflow(user_id)

                # Create initial state (conversation_id is already validated)
                initial_state = await composer.create_initial_state(user_id, msg.conversation_id)  # type: ignore

                # Execute workflow with streaming
                async for event in composer.execute_workflow(
                    workflow, initial_state, stream=True
                ):
                    # Convert composer events to SSE format
                    if isinstance(event, dict):
                        # Handle different event types
                        event_type = event.get("event", "chunk")

                        if event_type == "on_llm_stream":
                            # Stream token from LLM
                            chunk = event.get("data", {}).get("chunk", {})
                            if chunk:
                                content = (
                                    chunk.get("content", "")
                                    if isinstance(chunk, dict)
                                    else str(chunk)
                                )
                                if content:
                                    yield f"data: {json.dumps({'content': content})}\n\n"
                        elif event_type == "on_chain_end":
                            # End of workflow
                            yield f"data: {json.dumps({'type': 'done'})}\n\n"
                        else:
                            # Other events - pass through
                            yield f"data: {json.dumps(event)}\n\n"
                    else:
                        # Handle raw string events
                        yield f"data: {json.dumps({'content': str(event)})}\n\n"

            except Exception as e:
                logger.error(f"Error in composer chat completion: {e}")
                error_data = json.dumps({"error": str(e), "type": "error"})
                yield f"data: {error_data}\n\n"

        return StreamingResponse(
            composer_chat_completion(),
            media_type="text/event-stream",
        )

    except Exception as e:  # noqa: BLE001
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
