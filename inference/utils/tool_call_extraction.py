"""
Shared utilities for tool call extraction and processing from messages.

This module consolidates tool call extraction logic used across multiple agents
and components, using the strongly-typed utilities from tool_call_types.
"""

from typing import List, Optional, Dict, Any
import json
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from utils.logging import llmmllogger

from models import (
    Message,
    MessageContent,
    MessageContentType,
    MessageRole,
    ToolCall,
    tool,
)
from .tool_call_types import (
    extract_tool_calls_as_models,
    has_tool_call_requests_as_models,
    tool_call_request_to_execution_result,
)

logger = llmmllogger.bind(component="tool_call_extraction")


def extract_tool_calls_from_langchain_message(message: BaseMessage) -> List[ToolCall]:
    """
    Extract tool call requests from a LangChain message with comprehensive error handling.

    This is the main function that should be used across all agents and components
    for consistent tool call extraction.

    Args:
        message: LangChain message to extract tool calls from

    Returns:
        List of validated ToolCall objects
    """
    try:
        # Use the strongly-typed utility function
        tool_calls = extract_tool_calls_as_models(message)

        if tool_calls:
            logger.debug(
                "Extracted tool call requests",
                tool_calls_count=len(tool_calls),
                tool_names=[tc.name for tc in tool_calls],
            )

        return tool_calls

    except Exception as e:
        logger.error(
            "Failed to extract tool call requests",
            error=str(e),
            message_type=getattr(message, "type", "unknown"),
        )
        return []


def has_tool_calls_in_langchain_message(message: BaseMessage) -> bool:
    """
    Check if a LangChain message has tool call requests.

    Args:
        message: LangChain message to check

    Returns:
        True if message has tool call requests, False otherwise
    """
    try:
        return has_tool_call_requests_as_models(message)
    except Exception as e:
        logger.error(
            "Failed to check for tool calls",
            error=str(e),
            message_type=getattr(message, "type", "unknown"),
        )
        return False


def extract_tool_calls_from_streaming_chunks(
    chunks: List[BaseMessage],
) -> List[ToolCall]:
    """
    Extract tool calls from streaming response chunks.

    Args:
        chunks: List of streaming response chunks

    Returns:
        List of ToolCall objects found in chunks
    """
    logger.debug(
        f"🔍 extract_tool_calls_from_streaming_chunks called with {len(chunks)} chunks"
    )

    tool_calls = []

    for i, chunk in enumerate(chunks):
        try:
            logger.debug(f"🔍 Processing chunk {i}: type={type(chunk)}")

            # Check if chunk has tool_calls attribute
            if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                logger.debug(f"🔍 Chunk {i} has {len(chunk.tool_calls)} tool_calls")
                for j, tc in enumerate(chunk.tool_calls):
                    logger.debug(f"🔍 Tool call {j}: type={type(tc)}")

            # Look for tool call data in various chunk formats
            extracted = extract_tool_calls_as_models(chunk)
            if extracted:
                logger.debug(f"✅ Extracted {len(extracted)} tool calls from chunk {i}")
            tool_calls.extend(extracted)

        except Exception as e:
            logger.warning(
                "Failed to extract tool call from streaming chunk",
                error=str(e),
                chunk_preview=str(chunk)[:200],
                chunk_type=type(chunk).__name__,
            )

    logger.debug(
        f"🔍 extract_tool_calls_from_streaming_chunks returning {len(tool_calls)} total tool calls"
    )
    return tool_calls


def extract_tool_calls_from_message_content(
    message_content: List[MessageContent],
) -> List[ToolCall]:
    """
    Extract tool calls from Message content items.

    Args:
        message_content: List of MessageContent objects

    Returns:
        List of ToolCall objects found in content
    """
    tool_calls = []

    for content in message_content:
        try:
            if content.type == MessageContentType.TOOL_CALL and content.text:
                # Parse JSON tool call data from text
                tool_call_data = json.loads(content.text)

                if isinstance(tool_call_data, dict) and "name" in tool_call_data:
                    tool_call = ToolCall(
                        name=tool_call_data.get("name", "unknown"),
                        success=tool_call_data.get("success", True),
                        args=tool_call_data.get("args", {}),
                        result_data=tool_call_data.get("result_data", {}),
                        execution_id=tool_call_data.get("execution_id", None),
                        error_message=tool_call_data.get("error_message", None),
                        execution_time_ms=tool_call_data.get("execution_time_ms", None),
                    )
                    tool_calls.append(tool_call)

        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(
                "Failed to parse tool call from message content",
                error=str(e),
                content_text=(content.text[:100] if content.text else ""),
            )

    return tool_calls


def create_tool_call_message_content(tool_call: ToolCall) -> MessageContent:
    """
    Create a MessageContent object from a ToolCall for inclusion in messages.

    Args:
        tool_call: ToolCall object to convert

    Returns:
        MessageContent with TOOL_CALL type containing JSON data
    """
    try:
        tool_call_dict = {
            "name": tool_call.name,
            "success": tool_call.success,
            "args": tool_call.args,
            "result_data": tool_call.result_data,
            "execution_id": tool_call.execution_id,
            "error_message": tool_call.error_message,
            "execution_time_ms": tool_call.execution_time_ms,
        }

        return MessageContent(
            type=MessageContentType.TOOL_CALL,
            text=json.dumps(tool_call_dict),
            url=None,
        )

    except Exception as e:
        logger.error(
            "Failed to create tool call message content",
            error=str(e),
            tool_call_name=tool_call.name,
        )
        # Return a minimal valid MessageContent
        return MessageContent(
            type=MessageContentType.TOOL_CALL,
            text=json.dumps({"name": tool_call.name, "success": False, "args": {}}),
            url=None,
        )
