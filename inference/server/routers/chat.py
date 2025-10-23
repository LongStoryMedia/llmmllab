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


def extract_thinking_from_content(content: str) -> tuple[str, str]:
    """
    Extract <think>...</think> blocks from content and return cleaned content and thinking.
    Also detects thinking content that might not be wrapped in tags but appears to be reasoning.
    """
    if not content or not isinstance(content, str):
        return content, ""
    
    # Extract explicit think content
    think_pattern = r'<think>(.*?)</think>'
    think_matches = re.findall(think_pattern, content, re.DOTALL)
    thinking = "\n\n".join(think_matches) if think_matches else ""
    
    # Remove think tags from content
    cleaned_content = re.sub(think_pattern, '', content, flags=re.DOTALL)
    
    # Detect thinking patterns without tags (common patterns that indicate reasoning)
    # Look for content that starts with reasoning indicators
    reasoning_patterns = [
        r"(?:^|\n)((?:Okay|Alright|Let me|I need to|First|The user|Looking at|Thinking about|Based on|Given that).*?)(?=\n\n|\n[A-Z]|\n\d+\.|\n-|\n\*|$)",
        r"(?:^|\n)((?:To answer|In order to|The question|This request|The task).*?)(?=\n\n|\n[A-Z]|\n\d+\.|\n-|\n\*|$)",
    ]
    
    for pattern in reasoning_patterns:
        reasoning_matches = re.findall(pattern, cleaned_content, re.DOTALL | re.MULTILINE)
        if reasoning_matches:
            for match in reasoning_matches:
                # Check if this looks like reasoning (contains certain keywords)
                if any(keyword in match.lower() for keyword in ['need to', 'should', 'will', 'can', 'might', 'let me', 'i should', 'to answer']):
                    thinking += ("\n\n" if thinking else "") + match.strip()
                    # Remove this reasoning from the main content
                    cleaned_content = cleaned_content.replace(match, "")
    
    # Clean up extra whitespace but preserve intentional spacing
    cleaned_content = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_content)
    cleaned_content = cleaned_content.strip()
    
    return cleaned_content, thinking.strip() if thinking else ""


def parse_tool_calls_from_content(content: str) -> tuple[str, list]:
    """
    Parse tool calls from content and return cleaned content and tool calls.
    Enhanced to handle various tool call formats including function-call syntax.
    """
    if not content or not isinstance(content, str):
        return content, []
    
    tool_calls = []
    cleaned_content = content
    
    # Pattern 1: <tool_call> and <function-call> tags with JSON
    pattern1 = r'<(?:tool_call|function-call)>\s*(\{.*?\})\s*</(?:tool_call|function-call)>'
    matches1 = re.finditer(pattern1, content, re.DOTALL)
    
    for i, match in enumerate(matches1):
        try:
            json_str = match.group(1)
            tool_call_data = json.loads(json_str)
            
            tool_call = {
                "tool_name": tool_call_data.get("name", ""),
                "execution_id": f"call_{len(tool_calls)}",
                "success": True,
                "args": tool_call_data.get("args", tool_call_data.get("arguments", {})),
                "result_data": {},
                "execution_time_ms": 0,
            }
            tool_calls.append(tool_call)
            
            # Remove this match from content
            cleaned_content = cleaned_content.replace(match.group(0), "")
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse tool call: {e}")
            continue
    
    # Pattern 2: Function call syntax without tags (from user's example)
    # Match: <function-call>{"name":"web_search","arguments":{"query":"..."}}
    pattern2 = r'<function-call>(\{[^}]*"name"\s*:\s*"[^"]+"\s*[^}]*\})'
    matches2 = re.finditer(pattern2, content, re.DOTALL)
    
    for match in matches2:
        try:
            json_str = match.group(1)
            tool_call_data = json.loads(json_str)
            
            tool_call = {
                "tool_name": tool_call_data.get("name", ""),
                "execution_id": f"call_{len(tool_calls)}",
                "success": True,
                "args": tool_call_data.get("args", tool_call_data.get("arguments", {})),
                "result_data": {},
                "execution_time_ms": 0,
            }
            tool_calls.append(tool_call)
            
            # Remove this match from content
            cleaned_content = cleaned_content.replace(match.group(0), "")
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse function call: {e}")
            continue
    
    # Clean up extra whitespace but don't use strip() to preserve token spacing
    cleaned_content = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_content)
    cleaned_content = re.sub(r'^\s+|\s+$', '', cleaned_content)  # Only trim leading/trailing whitespace
    
    return cleaned_content, tool_calls


def extract_structured_data_from_events(events: list) -> dict:
    """
    Extract structured data from LangGraph workflow events.
    Returns thinking, tool_calls, analyses, and observer_messages.
    """
    thinking_parts = []
    tool_calls = []
    intent_analyses = []
    observer_messages = []
    tool_execution_results = {}  # Map execution_id to result data
    
    for event in events:
        if not isinstance(event, dict):
            continue
            
        event_type = event.get("event", "")
        event_data = event.get("data", {})
        
        # Extract thinking from chat model events - enhanced pattern detection
        if event_type == "on_chat_model_stream":
            chunk = event_data.get("chunk", {})
            if isinstance(chunk, dict):
                content = chunk.get("content", "")
                if content:
                    # Extract thinking from this chunk using enhanced function
                    _, think_content = extract_thinking_from_content(content)
                    if think_content:
                        thinking_parts.append(think_content)
        
        # Extract tool execution data
        elif event_type == "on_tool_start":
            tool_name = event_data.get("name", "")
            tool_input = event_data.get("input", {})
            execution_id = f"tool_{len(tool_calls)}"
            
            tool_call = {
                "tool_name": tool_name,
                "execution_id": execution_id,
                "success": True,
                "args": tool_input,
                "result_data": {},
                "execution_time_ms": 0,
            }
            tool_calls.append(tool_call)
            tool_execution_results[execution_id] = len(tool_calls) - 1  # Store index
            
        elif event_type == "on_tool_end":
            tool_name = event_data.get("name", "")
            tool_output = event_data.get("output", "")
            
            # Find corresponding tool call and update with result
            for i, tool_call in enumerate(tool_calls):
                if tool_call["tool_name"] == tool_name and not tool_call["result_data"]:
                    tool_call["result_data"] = {"output": tool_output}
                    break
        
        # Extract intent analyses from workflow state - enhanced extraction
        elif event_type == "on_chain_end":
            output = event_data.get("output", {})
            if isinstance(output, dict):
                # Check for intent classification in final state
                intent_classification = output.get("intent_classification", [])
                if intent_classification and isinstance(intent_classification, list):
                    intent_analyses.extend(intent_classification)
                
                # Also check for structured_response field which may contain intent analyses
                structured_response = output.get("structured_response", "")
                if structured_response and "intents=" in str(structured_response):
                    # This contains parsed intent analysis data
                    try:
                        # Extract the intent data from the structured response string
                        # Format: intents=[IntentAnalysis(...), ...]
                        import ast
                        # Simple parsing of the structured response
                        if "workflow_type=" in structured_response:
                            # This contains actual intent analysis data
                            logger.debug(f"Found structured intent analysis: {structured_response}")
                    except Exception as e:
                        logger.debug(f"Could not parse structured response: {e}")
        
        # Extract observer messages
        elif event_type == "on_chat_model_stream":
            chunk = event_data.get("chunk", {})
            if isinstance(chunk, dict):
                # Check if this is an observer message
                observer_msgs = chunk.get("observer_messages", [])
                if observer_msgs:
                    observer_messages.extend(observer_msgs)
    
    # Also look for intent analyses in chat model outputs that contain JSON
    for event in events:
        if event.get("event") == "on_chat_model_end":
            output = event.get("data", {}).get("output", {})
            content = ""
            if isinstance(output, dict):
                content = output.get("content", "")
            elif hasattr(output, "content"):
                content = str(output.content)
            
            # Check if the content contains intent analysis JSON
            if content and "intents" in content and "workflow_type" in content:
                try:
                    # Try to parse as JSON
                    analysis_data = json.loads(content)
                    if "intents" in analysis_data:
                        intent_analyses.extend(analysis_data["intents"])
                        logger.debug(f"Extracted {len(analysis_data['intents'])} intent analyses from model output")
                except json.JSONDecodeError:
                    # Content might contain JSON within other text
                    json_pattern = r'\{[^{}]*"intents"\s*:\s*\[[^\]]*\][^{}]*\}'
                    json_matches = re.findall(json_pattern, content)
                    for match in json_matches:
                        try:
                            analysis_data = json.loads(match)
                            if "intents" in analysis_data:
                                intent_analyses.extend(analysis_data["intents"])
                                logger.debug(f"Extracted {len(analysis_data['intents'])} intent analyses from embedded JSON")
                        except json.JSONDecodeError:
                            continue
    
    return {
        "thinking": "\n\n".join(thinking_parts) if thinking_parts else "",
        "tool_calls": tool_calls,
        "analyses": intent_analyses,
        "observer_messages": observer_messages,
    }


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
            try:
                thought_service = getattr(storage, 'thought', None)
                if thought_service:
                    thought_id = await thought_service.add_thought(
                        message_id=message_id,
                        text=thinking_content.strip()
                    )
                    if thought_id:
                        logger.debug(f"Stored thought {thought_id} for message {message_id}")
                    else:
                        logger.warning(f"Failed to store thought for message {message_id}")
            except Exception as e:
                logger.error(f"Error storing thought for message {message_id}: {e}")
        
        # Store intent analyses if present
        analyses = structured_data.get("analyses", [])
        if analyses:
            try:
                analysis_service = getattr(storage, 'analysis', None)
                if analysis_service:
                    for analysis in analyses:
                        try:
                            # Convert analysis to JSON format for storage
                            analysis_data = analysis if isinstance(analysis, dict) else {"analysis": str(analysis)}
                            
                            analysis_id = await analysis_service.add_analysis(
                                message_id=message_id,
                                analysis_data=analysis_data
                            )
                            if analysis_id:
                                logger.debug(f"Stored analysis {analysis_id} for message {message_id}")
                            else:
                                logger.warning(f"Failed to store analysis for message {message_id}")
                        except Exception as e:
                            logger.error(f"Error storing analysis for message {message_id}: {e}")
            except Exception as e:
                logger.error(f"Error accessing analysis service: {e}")
        
        # Store tool calls if present
        tool_calls = structured_data.get("tool_calls", [])
        if tool_calls:
            try:
                tool_call_service = getattr(storage, 'tool_call', None)
                if tool_call_service:
                    for tool_call in tool_calls:
                        try:
                            # Ensure tool_call is in proper format for storage
                            tool_data = tool_call if isinstance(tool_call, dict) else {"tool_call": str(tool_call)}
                            
                            tool_call_id = await tool_call_service.add_tool_call(
                                message_id=message_id,
                                tool_data=tool_data
                            )
                            if tool_call_id:
                                logger.debug(f"Stored tool call {tool_call_id} for message {message_id}")
                            else:
                                logger.warning(f"Failed to store tool call for message {message_id}")
                        except Exception as e:
                            logger.error(f"Error storing tool call for message {message_id}: {e}")
            except Exception as e:
                logger.error(f"Error accessing tool call service: {e}")
                    
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

                # Track all events for structured data extraction
                all_events = []
                accumulated_content = ""
                final_state = None
                current_thinking = ""

                # Execute workflow and stream events
                async for event in composer.execute_workflow(
                    workflow, initial_state, stream=True
                ):
                    if isinstance(event, dict):
                        all_events.append(event)  # Store all events for final processing
                        event_type = event.get("event", "")
                        event_data = event.get("data", {})

                        # Stream chat model content with proper parsing
                        if event_type == "on_chat_model_stream":
                            chunk = event_data.get("chunk", {})
                            content = ""

                            if isinstance(chunk, dict):
                                content = chunk.get("content", "")
                            elif hasattr(chunk, "content"):
                                content = str(chunk.content)

                            if content:
                                # Extract thinking and clean content for streaming
                                clean_content, thinking = extract_thinking_from_content(content)
                                
                                if thinking:
                                    current_thinking += thinking + "\n"
                                
                                if clean_content:
                                    # Remove tool calls from streaming content
                                    clean_content, _ = parse_tool_calls_from_content(clean_content)
                                    
                                    # Only stream actual response content, not reasoning or tool calls
                                    if clean_content and clean_content.strip():
                                        # Avoid streaming content that looks like JSON intent analysis
                                        if not (clean_content.strip().startswith('{') and 'intents' in clean_content):
                                            accumulated_content += clean_content
                                            
                                            chat_response = {
                                                "message": {
                                                    "role": "assistant",
                                                    "content": [{"type": "text", "text": clean_content}],
                                                },
                                                "thinking": current_thinking.strip() if current_thinking.strip() else None,
                                                "done": False,
                                            }
                                            yield f"{safe_json_serialize(chat_response)}\n"

                        elif event_type == "on_chain_end":
                            # Capture final state for response extraction
                            final_state = event_data.get("output", {})

                # Extract structured data from all collected events
                structured_data = extract_structured_data_from_events(all_events)
                
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
                    # Clean final content of thinking and tool calls for storage
                    clean_content, final_thinking = extract_thinking_from_content(final_content)
                    clean_content, final_tool_calls = parse_tool_calls_from_content(clean_content)
                    
                    # Use accumulated content if it's different and better than final content
                    if accumulated_content and len(accumulated_content) > len(clean_content):
                        clean_content = accumulated_content
                    
                    # Filter out JSON intent analysis from final content
                    if clean_content and clean_content.strip().startswith('{') and 'intents' in clean_content:
                        clean_content = ""  # Don't include JSON in final response
                    
                    # Combine thinking from streaming and final content
                    combined_thinking = current_thinking.strip()
                    if final_thinking and final_thinking not in combined_thinking:
                        combined_thinking += "\n\n" + final_thinking if combined_thinking else final_thinking
                    
                    # Merge tool calls from content parsing with those from events
                    all_tool_calls = structured_data["tool_calls"] + final_tool_calls
                    
                    # Create final response with structured data
                    final_response = {
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": clean_content}] if clean_content.strip() else [{"type": "text", "text": "Response completed successfully."}],
                        },
                        "done": True,
                    }
                    
                    # Add structured data fields
                    if combined_thinking.strip():
                        final_response["thinking"] = combined_thinking.strip()
                    
                    if all_tool_calls:
                        final_response["tool_calls"] = all_tool_calls
                    
                    if structured_data["analyses"]:
                        final_response["analyses"] = structured_data["analyses"]
                    
                    if structured_data["observer_messages"]:
                        final_response["observer_messages"] = structured_data["observer_messages"]
                    
                    # Save assistant response to database
                    if storage.message:
                        try:
                            # Use clean content for storage (no duplication)
                            storage_content = clean_content if clean_content.strip() else "Response completed successfully."
                            
                            assistant_message = Message(
                                conversation_id=conversation_id,
                                role=MessageRole.ASSISTANT,
                                content=[
                                    MessageContent(
                                        type=MessageContentType.TEXT,
                                        text=storage_content,
                                    )
                                ],
                            )
                            assistant_message_id = await storage.message.add_message(assistant_message)
                            logger.debug(
                                f"Assistant response stored for conversation {conversation_id}"
                            )
                            
                            # Store structured data in separate tables if message was saved successfully
                            if assistant_message_id:
                                await store_structured_response_data(
                                    assistant_message_id,
                                    combined_thinking,
                                    {
                                        "tool_calls": all_tool_calls,
                                        "analyses": structured_data["analyses"],
                                        "observer_messages": structured_data["observer_messages"],
                                    }
                                )
                            
                            # Update conversation title if one was generated by the workflow
                            if final_state and isinstance(final_state, dict):
                                generated_title = final_state.get("title")
                                if generated_title and storage.conversation:
                                    try:
                                        await storage.conversation.update_conversation_title(
                                            conversation_id, generated_title
                                        )
                                        logger.debug(
                                            f"Updated conversation {conversation_id} title to: {generated_title}"
                                        )
                                    except Exception as title_error:
                                        logger.warning(
                                            f"Failed to update conversation title: {title_error}"
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
