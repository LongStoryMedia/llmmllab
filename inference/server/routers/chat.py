"""
Simplified Chat router that delegates to composer interface.
All chat logic has been moved to the composer module for clean architectural separation.

Note: This router is included in app.py with both non-versioned and versioned paths:
- Non-versioned: /chat/...
- Versioned: /v1/chat/...
"""

import json
from typing import AsyncGenerator, Any, Dict, List
from typing_extensions import TypedDict
from pydantic import BaseModel

from langchain_core.runnables import RunnableConfig

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from server.middleware.auth import get_request_id, get_user_id, is_admin
from server.config import logger  # Import logger from config
from server.streaming_response_state import StreamingResponseState
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

                # Execute workflow with state and checkpointing
                events = []
                final_response_data = {}
                tool_execution_results = []  # Collect real tool execution results

                # Configure threading for persistent state management
                config = RunnableConfig(
                    configurable={
                        "thread_id": str(
                            conversation_id
                        ),  # Use conversation_id as thread for checkpointing
                        "checkpoint_ns": "",
                    }
                )

                async for event in workflow.astream_events(
                    initial_state,
                    config=config,
                    version="v2",
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

                    # Debug: Log all event types to understand what's available
                    if event_type and "tool" in event_type.lower():
                        logger.debug(f"🔧 Tool-related event: {event_type}")

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

                            if content:
                                # Use streaming state manager to process chunk
                                chat_response = streaming_state.process_chunk(content)
                                # logger.debug(
                                #     f"Stream content: '{serialize_event_data(chat_response)}'"
                                # )
                                yield f"{chat_response.model_dump_json(exclude_none=True)}\n"

                                # Let streaming state manage content accumulation
                                # Don't duplicate accumulation here

                    elif event_type == "on_tool_end":
                        # Capture actual tool execution results with real status and timing
                        try:
                            # Get tool execution details from event data
                            tool_name = event_data.get("name", "unknown_tool")
                            tool_input = event_data.get("input", {})
                            tool_output = event_data.get("output", {})
                            
                            # Calculate execution time from event timestamps if available
                            execution_time_ms = None
                            if isinstance(event, dict):
                                start_time = event.get("metadata", {}).get("start_time")
                                end_time = event.get("metadata", {}).get("end_time") 
                                if start_time and end_time:
                                    execution_time_ms = (end_time - start_time) * 1000

                            # Determine success based on whether we got an error
                            error_message = None
                            success = True
                            if isinstance(tool_output, dict) and "error" in tool_output:
                                success = False
                                error_message = str(tool_output.get("error", "Unknown error"))
                            elif hasattr(tool_output, "error"):
                                success = False
                                error_message = str(getattr(tool_output, "error", "Unknown error"))

                            # Filter out ToolRuntime and other non-serializable objects from args
                            clean_args = {}
                            if isinstance(tool_input, dict):
                                filtered_keys = []
                                for k, v in tool_input.items():
                                    if k == "runtime" or str(type(v).__name__) == "ToolRuntime":
                                        filtered_keys.append(k)
                                    else:
                                        clean_args[k] = v
                                
                                if filtered_keys:
                                    logger.debug(f"🔧 Filtered out non-serializable keys from tool args: {filtered_keys}")
                            else:
                                clean_args = {"input": str(tool_input)}

                            # Create real tool execution result
                            real_tool_result = ToolCall(
                                name=tool_name,
                                execution_id=f"tool_exec_{len(tool_execution_results)}",
                                success=success,
                                args=clean_args,
                                result_data=tool_output if isinstance(tool_output, dict) else {"output": str(tool_output)},
                                error_message=error_message,
                                execution_time_ms=execution_time_ms,
                            )
                            
                            tool_execution_results.append(real_tool_result)
                            
                            logger.info(
                                f"📊 Captured real tool execution: {tool_name} - "
                                f"Success: {success}, Time: {execution_time_ms}ms"
                            )
                            
                        except Exception as e:
                            logger.warning(f"Failed to capture tool execution result: {e}")

                    elif event_type.endswith("_end"):
                        # Debug: Log all end events to see what we get
                        if "tool" in event_type.lower():
                            logger.info(f"🛠️ Tool end event detected: {event_type}")
                            logger.info(f"   Event data keys: {list(event_data.keys()) if isinstance(event_data, dict) else 'not dict'}")
                        
                        # Capture final workflow data
                        if isinstance(event_data, dict):
                            output = event_data.get("output", {})
                        else:
                            output = getattr(event_data, "output", {})

                        logger.debug(
                            f"Chain end output: {serialize_event_data(output)}"
                        )
                        if output:
                            final_response_data = output

                # Get final consolidated response from streaming state
                final_response = streaming_state.get_final_response()
                
                # Replace streaming state tool calls with real execution results if available
                if tool_execution_results and final_response.message:
                    final_response.message.tool_calls = tool_execution_results
                    logger.info(f"🔄 Replaced streaming tool calls with {len(tool_execution_results)} real execution results")

                # Store the assistant's response in database using streaming state's accumulated content
                # Convert analyses from final response data to IntentAnalysis objects
                analyses = []
                raw_analyses = final_response_data.get("analyses", [])
                for analysis in raw_analyses:
                    if isinstance(analysis, IntentAnalysis):
                        analyses.append(analysis)
                    elif isinstance(analysis, dict):
                        try:
                            analyses.append(IntentAnalysis(**analysis))
                        except Exception as e:
                            logger.warning(
                                f"Failed to convert analysis dict to IntentAnalysis: {e}"
                            )

                # Debug: Check if response_buffer is empty
                if not streaming_state.response_buffer.strip():
                    logger.warning(
                        f"⚠️ Empty response_buffer detected! Accumulated thinking: {len(streaming_state.accumulated_thinking)} chars, Tool calls: {len(streaming_state.tool_calls)}"
                    )
                else:
                    logger.info(
                        f"✅ Response buffer has {len(streaming_state.response_buffer)} characters"
                    )

                # Use real tool execution results if available, otherwise fall back to streaming state
                final_tool_calls = None
                if tool_execution_results:
                    # Use real tool execution results with actual status and timing
                    final_tool_calls = tool_execution_results
                    logger.info(f"✅ Using {len(tool_execution_results)} real tool execution results")
                elif streaming_state.tool_calls:
                    # Fallback to streaming state tool calls (legacy behavior)
                    final_tool_calls = streaming_state.tool_calls
                    logger.warning(f"⚠️ Using {len(streaming_state.tool_calls)} streaming state tool calls (no real results)")

                assistant_message = Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=streaming_state.response_buffer,
                        )
                    ],
                    conversation_id=conversation_id,
                    tool_calls=final_tool_calls,
                    thoughts=(
                        [Thought(text=streaming_state.accumulated_thinking)]
                        if streaming_state.accumulated_thinking
                        and streaming_state.accumulated_thinking.strip()
                        else None
                    ),
                    analyses=analyses if analyses else None,
                )

                try:
                    message_result = await storage.get_service(
                        storage.message
                    ).add_message(assistant_message)
                    # Handle case where add_message returns just the ID as an integer
                    if isinstance(message_result, int):
                        assistant_message_id = message_result
                    elif isinstance(message_result, dict):
                        assistant_message_id = message_result.get("id")
                    else:
                        assistant_message_id = None

                    if assistant_message_id:
                        # Check for generated todos from planning middleware
                        generated_todos = final_response_data.get("generated_todos", [])
                        if generated_todos:
                            logger.info(
                                f"Intent analysis generated {len(generated_todos)} todos"
                            )

                            # Add todo notification to response
                            todo_count = len(generated_todos)
                            todo_message = f"✅ Generated {todo_count} todo{'s' if todo_count != 1 else ''} based on your request."

                            # Update the streaming state response buffer to include todo notification
                            if streaming_state.response_buffer:
                                streaming_state.response_buffer += f"\n\n{todo_message}"
                            else:
                                streaming_state.response_buffer = todo_message

                        # Only store analyses separately as they're not handled by add_message
                        # Thoughts and tool_calls are already stored via add_message
                        analyses = []
                        raw_analyses = final_response_data.get("analyses", [])
                        for analysis in raw_analyses:
                            if isinstance(analysis, IntentAnalysis):
                                analyses.append(analysis)
                            elif isinstance(analysis, dict):
                                try:
                                    analyses.append(IntentAnalysis(**analysis))
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to convert analysis dict to IntentAnalysis: {e}"
                                    )

                        # Store only analyses separately - thoughts and tool_calls already stored
                        if analyses:
                            for analysis in analyses:
                                if isinstance(analysis, IntentAnalysis):
                                    await storage.get_service(
                                        storage.analysis
                                    ).add_analysis(
                                        message_id=assistant_message_id,
                                        intent_analysis=analysis,
                                    )
                            logger.info(
                                f"Stored {len(analyses)} intent analyses for message {assistant_message_id}"
                            )

                except Exception as storage_error:
                    logger.error(f"Failed to store assistant message: {storage_error}")

                # Update final response with any todo notifications
                if final_response.message and final_response.message.content:
                    # Update the message content to reflect the updated streaming state buffer
                    final_response.message.content[0].text = (
                        streaming_state.response_buffer
                    )

                # Yield final response with done=True
                final_response.done = True
                final_json = safe_json_serialize(final_response.dict(exclude_none=True))
                yield f"{final_json}\n"

            except Exception as workflow_error:
                logger.error(
                    f"Error in composer workflow: {workflow_error}", exc_info=True
                )

                # Create error response
                error_response = ChatResponse(
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT,
                                text=f"I encountered an error processing your request: {str(workflow_error)}",
                            )
                        ],
                    ),
                    done=True,
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
