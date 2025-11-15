"""
Simplified Chat router that delegates to composer interface.
All chat logic has been moved to the composer module for clean architectural separation.

Note: This router is included in app.py with both non-versioned and versioned paths:
- Non-versioned: /chat/...
- Versioned: /v1/chat/...
"""

import json
from typing import AsyncGenerator, Any, AsyncIterator, List, cast
from typing_extensions import TypedDict

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.schema import StandardStreamEvent
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from server.middleware.auth import get_request_id, get_user_id, is_admin
from server.config import logger  # Import logger from config
from db import storage  # Import database storage
from models import (
    MessageRole,
    MessageContent,
    MessageContentType,
    ChatResponse,
    Message,
    ToolCall,
    Thought,
    IntentAnalysis,
)
from utils.logging import serialize_event_data  # Import logging utility

# Import composer interface and streaming state management
import composer
from runner import ReasoningAwareAIMessageChunk

router = APIRouter(prefix="/chat", tags=["chat"])


class StructuredResponseData(TypedDict):
    """Strongly typed structure for response data storage."""

    thoughts: List[Thought]
    tool_calls: List[ToolCall]
    analyses: List[IntentAnalysis]


async def store_structured_response_data(
    message_id: int, structured_data: StructuredResponseData
) -> None:
    """
    Store structured response data (thoughts, analyses, tool_calls) in the database.

    Args:
        message_id: The ID of the assistant message
        structured_data: Strongly typed dictionary containing thoughts, tool_calls and analyses
    """
    try:
        # Store thoughts if present
        thoughts = structured_data.get("thoughts", [])
        if thoughts:
            for thought in thoughts:
                if isinstance(thought, Thought):
                    thought_obj = Thought(
                        message_id=message_id,
                        text=thought.text,
                    )
                    await storage.get_service(storage.thought).add_thought(thought_obj)
            logger.info(f"Stored {len(thoughts)} thoughts for message {message_id}")

        # Store intent analyses if present
        analyses = structured_data.get("analyses", [])
        if analyses:
            for analysis in analyses:
                if isinstance(analysis, IntentAnalysis):
                    await storage.get_service(storage.analysis).add_analysis(
                        message_id=message_id,
                        intent_analysis=analysis,
                    )
            logger.info(
                f"Stored {len(analyses)} intent analyses for message {message_id}"
            )

        # Store tool calls if present
        tool_calls = structured_data.get("tool_calls", [])
        if tool_calls:
            for tool_call in tool_calls:
                if isinstance(tool_call, ToolCall):
                    await storage.get_service(storage.tool_call).add_tool_call(
                        tool_call=tool_call,
                    )
            logger.info(
                f"Stored {len(tool_calls)} tool execution results for message {message_id}"
            )

        logger.info(f"Structured response data stored for message {message_id}")

    except Exception as e:
        logger.error(
            f"Failed to store structured response data for message {message_id}: {e}"
        )


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
        # Store the user message in database first (with fallback for connection issues)
        await storage.get_service(storage.message).add_message(msg)
        # Capture variables for the async generator
        conversation_id = msg.conversation_id

        # Direct composer workflow orchestration
        async def composer_chat_completion() -> AsyncIterator[str]:
            """Handle chat completions by delegating to composer interface."""
            await composer.initialize_composer()

            # Compose workflow for user
            workflow = await composer.compose_workflow(user_id)

            # Create initial state (conversation_id is already validated)
            initial_state = await composer.create_initial_state(
                user_id, conversation_id
            )

            async for event in composer.execute_workflow(initial_state, workflow):
                yield f"{event.model_dump_json()}"

        return StreamingResponse(
            composer_chat_completion(),
            media_type="application/json",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffer
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


def safe_json_serialize(obj: Any) -> str:
    """Safely serialize objects to JSON, handling non-serializable types."""

    def json_serializer(obj):
        if isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        elif hasattr(obj, "dict") and callable(obj.dict):
            return obj.dict()
        else:
            return str(obj)

    try:
        return json.dumps(obj, default=json_serializer, ensure_ascii=False)
    except Exception as e:
        # If all else fails, return a safe error representation
        return json.dumps(
            {
                "error": f"Serialization failed: {str(e)}",
                "original_type": str(type(obj)),
            }
        )
