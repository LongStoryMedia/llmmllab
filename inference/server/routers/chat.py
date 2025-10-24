"""
Simplified Chat router that delegates to composer interface.
All chat logic has been moved to the composer module for clean architectural separation.

Note: This router is included in app.py with both non-versioned and versioned paths:
- Non-versioned: /chat/...
- Versioned: /v1/chat/...
"""

import json
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

# Import composer interface and streaming state management
import composer
from server.streaming_response_state import StreamingResponseState

router = APIRouter(prefix="/chat", tags=["chat"])


async def store_structured_response_data(
    message_id: int,
    thinking_content: str,
    structured_data: dict
) -> None:
    """
    Store structured response data (thoughts, analyses, tool_calls) in the database.
    
    Args:
        message_id: The ID of the assistant message
        thinking_content: The combined thinking/reasoning content
        structured_data: Dictionary containing tool_calls and analyses
    """
    try:
        # Store thinking content if present
        if thinking_content and thinking_content.strip():
            await storage.get_service(storage.thought).add_thought(
                message_id=message_id,
                text=thinking_content.strip(),
            )
            logger.info(f"Stored thinking content for message {message_id}")
        
        # Store intent analyses if present
        analyses = structured_data.get("analyses", [])
        if analyses:
            for analysis in analyses:
                if isinstance(analysis, dict) and analysis:
                    await storage.get_service(storage.intent_analysis).add_analysis(
                        message_id=message_id,
                        workflow_type=analysis.get("workflow_type", "general"),
                        complexity_level=analysis.get("complexity_level", "SIMPLE"),
                        required_capabilities=analysis.get("required_capabilities", []),
                        domain_specificity=analysis.get("domain_specificity", 0.0),
                        reusability_potential=analysis.get("reusability_potential", 0.0),
                        confidence=analysis.get("confidence", 0.0),
                        technical_domain=analysis.get("technical_domain"),
                        response_format=analysis.get("response_format"),
                        tool_complexity_score=analysis.get("tool_complexity_score", 0.0),
                        computational_requirements=analysis.get("computational_requirements", "LOW"),
                    )
            logger.info(f"Stored {len(analyses)} intent analyses for message {message_id}")
        
        # Store tool calls if present
        tool_calls = structured_data.get("tool_calls", [])
        if tool_calls:
            for tool_call in tool_calls:
                if isinstance(tool_call, dict) and tool_call.get("tool_name"):
                    await storage.get_service(storage.tool_execution_result).add_tool_execution_result(
                        message_id=message_id,
                        tool_name=tool_call["tool_name"],
                        execution_id=tool_call.get("execution_id", ""),
                        success=tool_call.get("success", False),
                        args=tool_call.get("args", {}),
                        result_data=tool_call.get("result_data", {}),
                        error_message=tool_call.get("error_message"),
                        execution_time_ms=tool_call.get("execution_time_ms", 0.0),
                    )
            logger.info(f"Stored {len(tool_calls)} tool execution results for message {message_id}")
                    
        logger.info(f"Structured response data stored for message {message_id}")
        
    except Exception as e:
        logger.error(f"Failed to store structured response data for message {message_id}: {e}")


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

                # Initialize streaming response state manager
                streaming_state = StreamingResponseState()

                # Execute workflow with state
                events = []
                final_response_data = {}

                async for event in workflow.astream_events(
                    initial_state, version="v2"
                ):
                    events.append(event)
                    
                    # Handle both dict and object event formats
                    if isinstance(event, dict):
                        event_type = event.get("event", "")
                        event_data = event.get("data", {})
                    else:
                        # For non-dict events, try to get event type from attributes
                        event_type = getattr(event, "event", "")
                        event_data = getattr(event, "data", {})

                    # Debug logging for events
                    logger.debug(f"Processing event: {event_type}, event_data: {event_data}")

                    # Process streaming events for immediate response
                    if event_type == "on_chat_model_stream":
                        # Handle streaming tokens
                        chunk = event.get("data", {}).get("chunk")
                        if chunk:
                            # Extract content from chunk  
                            if hasattr(chunk, "content"):
                                content = chunk.content
                            elif isinstance(chunk, dict):
                                content = chunk.get("content", "")
                            else:
                                content = str(chunk) if chunk else ""
                            
                            logger.debug(f"Stream content: '{content}'")
                            
                            if content:
                                # Use streaming state manager to process chunk
                                chat_response = streaming_state.process_chunk(content)
                                
                                # Only yield if there's actual content to send
                                if (chat_response.message and chat_response.message.content) or chat_response.thinking or chat_response.tool_calls:
                                    response_json = safe_json_serialize(chat_response.dict(exclude_none=True))
                                    yield f"{response_json}\n"
                                
                                # Let streaming state manage content accumulation
                                # Don't duplicate accumulation here

                    elif event_type == "on_chat_model_end":
                        # Skip this event to prevent duplicate processing
                        # Content is already handled in streaming events
                        logger.debug(f"Model end - skipping to prevent duplication")
                        pass
                        
                    elif event_type == "on_chain_end":
                        # Capture final workflow data
                        if isinstance(event_data, dict):
                            output = event_data.get("output", {})
                        else:
                            output = getattr(event_data, "output", {})
                            
                        logger.debug(f"Chain end output: {output}")
                        if output:
                            final_response_data = output

                # Get final consolidated response from streaming state
                final_response = streaming_state.get_final_response()

                # Store the assistant's response in database using streaming state's accumulated content
                assistant_message = Message(
                    role=MessageRole.ASSISTANT,
                    content=[MessageContent(type=MessageContentType.TEXT, text=streaming_state.response_buffer)],
                    conversation_id=conversation_id,
                )

                try:
                    message_result = await storage.get_service(storage.message).add_message(assistant_message)
                    # Handle case where add_message returns just the ID as an integer
                    if isinstance(message_result, int):
                        assistant_message_id = message_result
                    elif isinstance(message_result, dict):
                        assistant_message_id = message_result.get("id")
                    else:
                        assistant_message_id = None

                    if assistant_message_id:
                        # Store structured data (thinking, tool calls, analyses)
                        structured_data = {
                            "tool_calls": streaming_state.tool_calls,
                            "analyses": final_response_data.get("analyses", []),
                        }
                        
                        await store_structured_response_data(
                            assistant_message_id,
                            streaming_state.accumulated_thinking,
                            structured_data
                        )

                except Exception as storage_error:
                    logger.error(f"Failed to store assistant message: {storage_error}")

                # Yield final response with done=True
                final_response.done = True
                final_json = safe_json_serialize(final_response.dict(exclude_none=True))
                yield f"{final_json}\n"

            except Exception as workflow_error:
                logger.error(f"Error in composer workflow: {workflow_error}", exc_info=True)
                
                # Create error response
                error_response = ChatResponse(
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content=[MessageContent(
                            type=MessageContentType.TEXT,
                            text=f"I encountered an error processing your request: {str(workflow_error)}"
                        )]
                    ),
                    done=True
                )
                
                error_json = safe_json_serialize(error_response.dict(exclude_none=True))
                yield f"{error_json}\n"

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