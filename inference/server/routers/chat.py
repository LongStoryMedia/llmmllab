"""
Chat router for handling conversations and completions.
This implementation uses LangChain for enhanced RAG capabilities.
Updated to use pipeline factory and intent-based service instantiation.

Note: This router is included in app.py with both non-versioned and versioned paths:
- Non-versioned: /chat/...
- Versioned: /v1/chat/...
"""

import asyncio
from datetime import datetime as dt
from typing import Any, Coroutine

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from models import ChatResponse, Conversation, Message, MessageContentType, MessageRole
from server.auth import get_request_id, get_user_id, is_admin
from server.config import logger  # Import logger from config
from server.db import storage  # Import database storage
from server.services.completion import agent_chat_completion
from server.services.context import get_conversation_context_from_request

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/completions", response_model=ChatResponse)
async def chat_completion(
    msg: Message, request: Request, background_tasks: BackgroundTasks
):
    """
    Handle chat completions by processing a single user message and generating a response.
    This implementation uses LangChain for enhanced RAG capabilities including:

    1. Document retrieval from PostgreSQL vector store
    2. Web search integration (only when intent.web_search is True)
    3. URL content extraction
    4. Reranking of retrieved documents
    5. Deduplication of information
    6. Context-aware summarization
    7. Enhanced prompt creation with retrieved contexts
    8. Streaming or complete response generation
    """
    # Set up request context information
    user_id = get_user_id(request)
    request_id = get_request_id(request)
    assert user_id, f"User ID not found for request {request_id}"
    assert msg.conversation_id, f"Conversation ID not found for request {request_id}"

    conversation_ctx = await get_conversation_context_from_request(
        request, msg.conversation_id
    )
    assert (
        conversation_ctx.conversation.id >= 0
    ), f"Invalid conversation ID {conversation_ctx.conversation.id} for request {request_id}"
    # Verify message is from user - the only validation we still need
    assert msg, f"User message not found for request {request_id}"
    assert (
        msg.role == MessageRole.USER
    ), f"Invalid message role {msg.role} for request {request_id}"
    # Log the start of request processing
    logger.info(f"Processing chat completion request {request_id} for user {user_id}")

    user_cfg = await storage.get_service(storage.user_config).get_user_config(user_id)
    assert user_cfg, f"User config not found for user {user_id}"

    # Process user message with enhanced RAG
    try:
        # Add message - also sets intent
        embeddings, message_id = await conversation_ctx.add_message(msg)
        if not embeddings:
            logger.warning(f"Empty embedding vector for message {message_id}")

        summarization_task = conversation_ctx.summary_context.summarize(
            conversation_ctx.messages
        )
        query = next(
            (
                c.text
                for c in msg.content
                if c.type == MessageContentType.TEXT and c.text
            ),
            "",
        )
        assert conversation_ctx.intent, "Intent not set in conversation context"
        memory_task = (
            conversation_ctx.memory_context.retrieve_memories(embeddings)
            if conversation_ctx.intent.memory and query
            else None
        )
        web_task = (
            conversation_ctx.search_context.search(
                msg, conversation_ctx.conversation.id
            )
            if conversation_ctx.intent.web_search and msg
            else None
        )

        # Create a list for asyncio.gather
        tasks: list[Coroutine[Any, Any, Any]] = [summarization_task]
        if memory_task:
            tasks.append(memory_task)
        if web_task:
            tasks.append(web_task)
        await asyncio.gather(*tasks)

        if len(
            conversation_ctx.messages
        ) == 1 or conversation_ctx.conversation.title.startswith("New Conversation"):
            await conversation_ctx.generate_title()

        # Use enhanced chat completion logic which determines whether to use agentic workflow
        return StreamingResponse(
            agent_chat_completion(
                conversation_ctx=conversation_ctx,
                background_tasks=background_tasks,
            ),
            media_type="text/event-stream",
        )

    except Exception as e:  # noqa: BLE001
        logger.error(f"Error in chat completion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in chat completion: {str(e)}",
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
