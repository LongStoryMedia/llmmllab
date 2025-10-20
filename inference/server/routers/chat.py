"""
Simplified Chat router that delegates to composer interface.
All chat logic has been moved to the composer module for clean architectural separation.

Note: This router is included in app.py with both non-versioned and versioned paths:
- Non-versioned: /chat/...
- Versioned: /v1/chat/...
"""

import json
import os
import re
from typing import AsyncGenerator, Any, Dict

from langchain_core.runnables.schema import StandardStreamEvent, CustomStreamEvent

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
)

# Import composer interface
import composer

router = APIRouter(prefix="/chat", tags=["chat"])


def parse_tool_calls_from_content(content: str) -> tuple[str, list]:
    """
    Parse tool calls from content and return cleaned content and tool calls.
    Adapted from LlamaCpp pipeline tool call parsing.
    """
    if not content or not isinstance(content, str):
        return content, []
    
    # Regex pattern to match both <tool_call> and <function-call> tags
    pattern = r'<(?:tool_call|function-call)>\s*(\{.*?\})\s*</(?:tool_call|function-call)>'
    
    tool_calls = []
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for i, match in enumerate(matches):
        try:
            json_str = match.group(1)
            tool_call_data = json.loads(json_str)
            
            # Convert to LangChain tool call format
            tool_call = {
                "id": f"call_{i}",
                "name": tool_call_data.get("name", ""),
                "args": tool_call_data.get("args", tool_call_data.get("arguments", {})),
                "type": "tool_call"
            }
            tool_calls.append(tool_call)
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse tool call: {e}")
            continue
    
    # Remove tool call tags from content
    cleaned_content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    # Also remove <think> tags and their contents
    cleaned_content = re.sub(r'<think>.*?</think>', '', cleaned_content, flags=re.DOTALL)
    
    # Clean up extra whitespace
    cleaned_content = re.sub(r'\n\s*\n', '\n\n', cleaned_content).strip()
    
    return cleaned_content, tool_calls


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
        async def composer_chat_completion() -> AsyncGenerator[str, None]:
            """Handle chat completions by delegating to composer interface."""
            try:
                # Initialize composer service if needed
                await composer.initialize_composer()

                # Compose workflow for user
                workflow = await composer.compose_workflow(user_id)

                # Create initial state (conversation_id is already validated)
                initial_state = await composer.create_initial_state(
                    user_id, conversation_id
                )

                # Track accumulated content for final response
                accumulated_content = ""
                final_state = None

                # Execute workflow and stream events
                async for event in composer.execute_workflow(
                    workflow, initial_state, stream=True
                ):
                    if isinstance(event, dict):
                        event_type = event.get("event", "")
                        event_data = event.get("data", {})

                        # Only stream from chat model events to avoid duplication
                        if event_type == "on_chat_model_stream":
                            chunk = event_data.get("chunk", {})
                            content = ""

                            if isinstance(chunk, dict):
                                content = chunk.get("content", "")
                            elif hasattr(chunk, "content"):
                                content = str(chunk.content)

                            if content:
                                accumulated_content += content
                                
                                # Stream raw content chunks (tool calls will be parsed at the end)
                                chat_response = {
                                    "message": {
                                        "role": "assistant",
                                        "content": [{"type": "text", "text": content}],
                                    },
                                    "done": False,
                                }
                                yield f"{safe_json_serialize(chat_response)}\n"

                        elif event_type == "on_chain_end":
                            # Capture final state for response extraction
                            final_state = event_data.get("output", {})

                # Extract final response from workflow state or accumulated content
                final_content = ""
                
                if final_state and isinstance(final_state, dict):
                    messages = final_state.get("messages", [])
                    if messages and isinstance(messages, list) and len(messages) > 0:
                        # Get the last assistant message
                        last_message = None
                        for msg in reversed(messages):
                            if isinstance(msg, dict):
                                role = msg.get("type", "").lower()  # LangChain message type
                                if role == "ai" or role == "assistant":
                                    last_message = msg
                                    break
                            elif hasattr(msg, "type") and str(msg.type).lower() in [
                                "ai",
                                "aimessage",
                            ]:
                                last_message = msg
                                break

                        if last_message:
                            # Extract content from LangChain message
                            if isinstance(last_message, dict):
                                final_content = last_message.get("content", "")
                            elif hasattr(last_message, "content"):
                                final_content = str(last_message.content)
                
                # Use accumulated content if no final state content
                if not final_content and accumulated_content:
                    final_content = accumulated_content

                if final_content:
                    # Parse tool calls from the final content
                    cleaned_content, tool_calls = parse_tool_calls_from_content(final_content)
                    
                    # Create final response with parsed tool calls
                    final_response = {
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": cleaned_content}],
                        },
                        "done": True,
                    }
                    
                    # Add tool calls if any were found
                    if tool_calls:
                        final_response["message"]["tool_calls"] = tool_calls
                    
                    # Save assistant response to database
                    if storage.message:
                        try:
                            assistant_message = Message(
                                conversation_id=conversation_id,
                                role=MessageRole.ASSISTANT,
                                content=[
                                    MessageContent(
                                        type=MessageContentType.TEXT,
                                        text=cleaned_content,
                                    )
                                ],
                            )
                            await storage.message.add_message(assistant_message)
                            logger.debug(
                                f"Assistant response stored for conversation {conversation_id}"
                            )
                        except Exception as storage_error:
                            logger.warning(
                                f"Failed to store assistant response: {storage_error}"
                            )

                    yield f"{safe_json_serialize(final_response)}\n"
                    return

                # Fallback if no response was extracted
                fallback_response = {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Response completed successfully."}
                        ],
                    },
                    "done": True,
                }
                yield f"{safe_json_serialize(fallback_response)}\n"

            except Exception as e:
                logger.error(f"Error in composer chat completion: {e}")
                error_data = safe_json_serialize({"error": str(e), "type": "error"})
                yield f"{error_data}\n"
            finally:
                # Always send a final done event to signal stream completion
                yield f"{safe_json_serialize({'type': 'stream_end'})}\n"

        return StreamingResponse(
            composer_chat_completion(),
            media_type="application/json",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffer
            },
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


def safe_json_serialize(obj: Any) -> str:
    """Safely serialize objects to JSON, handling non-serializable types."""

    def json_serializer(obj):
        if isinstance(obj, set):
            return list(obj)  # Convert sets to lists
        elif hasattr(obj, "__dict__"):
            return obj.__dict__  # Convert objects to dicts
        elif hasattr(obj, "dict") and callable(obj.dict):
            return obj.dict()  # Handle Pydantic models
        else:
            return str(obj)  # Fallback to string representation

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
