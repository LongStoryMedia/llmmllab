"""
Simplified Chat router that delegates to composer interface.
All chat logic has been moved to the composer module for clean architectural separation.

Note: This router is included in app.py with both non-versioned and versioned paths:
- Non-versioned: /chat/...
- Versioned: /v1/chat/...
"""

from datetime import datetime as dt
from typing import AsyncGenerator
import json

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from models import ChatResponse, Conversation, Message, MessageContentType, MessageRole
from server.auth import get_request_id, get_user_id, is_admin
from server.config import logger  # Import logger from config
from db import storage  # Import database storage

# Import composer interface
import composer

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/completions", response_model=ChatResponse)
async def chat_completion(
    msg: Message, request: Request, background_tasks: BackgroundTasks
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
                async for event in composer.execute_workflow(workflow, initial_state, stream=True):
                    # Convert composer events to SSE format
                    if isinstance(event, dict):
                        # Handle different event types
                        event_type = event.get("event", "chunk")
                        
                        if event_type == "on_llm_stream":
                            # Stream token from LLM
                            chunk = event.get("data", {}).get("chunk", {})
                            if chunk:
                                content = chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
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
            error_detail = "Unable to create AI workflow. Please check your configuration."
        elif "unknown model architecture" in str(e):
            error_detail = "Model architecture not supported. Please try a different model."
        elif "Failed to create llama_context" in str(e):
            error_detail = "Model failed to load. This may be due to insufficient memory."

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


@router.get("/conversations", response_model=list[Conversation])
async def list_conversations(request: Request):
    """
    List all conversations for the user.
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Check if database is initialized
    if not storage.initialized or storage.conversation is None:
        logger.warning("Database not initialized, cannot list conversations")
        raise HTTPException(status_code=503, detail="Database service unavailable")

    try:
        # Get all conversations for the user
        conversations = await storage.conversation.get_user_conversations(user_id)
        return conversations
    except Exception as e:  # noqa: BLE001, justified for DB errors
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: int, request: Request):
    """
    Get a specific conversation by ID.
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Check if database is initialized
    if not storage.initialized or storage.conversation is None:
        logger.warning("Database not initialized, cannot get conversation")
        raise HTTPException(status_code=503, detail="Database service unavailable")

    try:
        # Get the conversation
        conversation = await storage.conversation.get_conversation(conversation_id)

        # Check if conversation exists
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Check if user has access to this conversation
        if conversation.user_id != user_id and not is_admin(request):
            raise HTTPException(
                status_code=403, detail="Access denied to this conversation"
            )

        return conversation
    except HTTPException as e:
        raise e
    except Exception as e:  # noqa: BLE001, justified for DB errors
        logger.error(f"Error getting conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e


@router.get("/conversations/{conversation_id}/messages", response_model=list[Message])
async def get_conversation_messages(conversation_id: int, request: Request):
    """
    Get messages for a specific conversation.
    This endpoint retrieves all messages for a given conversation_id and ensures they
    are properly formatted according to the Message schema with valid content arrays.
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Log the request for debugging
    logger.debug(
        f"Getting messages for conversation {conversation_id}, user: {user_id}"
    )

    # Check if database is initialized
    if not storage.initialized or not storage.conversation or not storage.message:
        logger.warning("Database not initialized, cannot get messages")
        raise HTTPException(status_code=503, detail="Database service unavailable")

    try:
        # First check if conversation exists and user has access
        conversation = await storage.conversation.get_conversation(conversation_id)

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Check if user has access to this conversation
        if conversation.user_id != user_id and not is_admin(request):
            raise HTTPException(
                status_code=403, detail="Access denied to this conversation"
            )

        # Get all messages for the conversation
        messages = await storage.message.get_messages_by_conversation_id(
            conversation_id, 500, 0
        )

        # Log retrieved messages for debugging
        logger.debug(
            f"Retrieved {len(messages)} messages for conversation {conversation_id}"
        )

        return messages or []

    except HTTPException as e:
        raise e
    except Exception as e:  # noqa: BLE001, justified for DB errors
        logger.error(f"Error fetching messages for conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int, request: Request):
    """
    Delete a conversation and all its messages.
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Check if database is initialized
    if not storage.initialized or storage.conversation is None:
        logger.warning("Database not initialized, cannot delete conversation")
        raise HTTPException(status_code=503, detail="Database service unavailable")

    try:
        # First check if conversation exists and user has access
        db_conversation = await storage.conversation.get_conversation(conversation_id)

        if not db_conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Check if user has access to this conversation
        if db_conversation.user_id != user_id and not is_admin(request):
            raise HTTPException(
                status_code=403, detail="Access denied to this conversation"
            )

        # Delete the conversation
        await storage.conversation.delete_conversation(conversation_id)

        return {
            "status": "success",
            "message": f"Conversation {conversation_id} deleted",
        }
    except HTTPException as e:
        raise e
    except Exception as e:  # noqa: BLE001, justified for DB errors
        logger.error(f"Error deleting conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e


@router.post("/conversations", response_model=Conversation)
async def create_conversation(request: Request):
    """
    Create a new conversation.
    """
    user_id = get_user_id(request)

    try:
        assert user_id, "User ID not found"
        convo = storage.get_service(storage.conversation)
        # Create the conversation in the database
        conversation_id = await convo.create_conversation(
            user_id, f"New conversation ({dt.now().strftime('%Y-%m-%d %H:%M')})"
        )

        if not conversation_id:
            raise HTTPException(status_code=500, detail="Failed to create conversation")

        # Get the newly created conversation
        return await convo.get_conversation(conversation_id)
    except HTTPException as e:
        raise e
    except Exception as e:  # noqa: BLE001, justified for DB errors
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e


@router.post("/pause/{conversation_id}")
async def pause_generation(conversation_id: int, request: Request):
    """
    Pause text generation for a conversation.
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not storage.conversation:
        raise HTTPException(status_code=503, detail="Conversation service unavailable")

    # Check if conversation exists and user has access
    try:
        conversation = await storage.conversation.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if conversation.user_id != user_id and not is_admin(request):
            raise HTTPException(
                status_code=403, detail="Access denied to this conversation"
            )
    except HTTPException as e:
        raise e
    except Exception as e:  # noqa: BLE001, justified for DB errors
        logger.error(f"Error validating conversation access: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e

    # Signal the generation to pause (implementation depends on your streaming setup)
    try:
        # This is a placeholder for actual pause implementation
        # In a real implementation, you might set a flag in a shared state,
        # send a signal to the generator, or use a pub/sub system
        return {"status": "success", "message": "Generation paused"}
    except Exception as e:
        logger.error(f"Error pausing generation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to pause generation: {str(e)}"
        ) from e
