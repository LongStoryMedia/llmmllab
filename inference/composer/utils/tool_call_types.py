"""
Strong type definitions for tool call handling.

This module provides clear, typed interfaces to eliminate confusion between:
- Tool call requests (what the AI wants to call)
- Tool execution results (what happened when we called it)
"""

from typing import Dict, Any, Optional, List, Union, TypedDict, Protocol
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    ToolMessage,
    ToolCall as LangChainToolCall,
)
from models import ToolCall


class ToolCallCapableMessage(Protocol):
    """Protocol for messages that can contain tool calls."""

    tool_calls: Optional[List[LangChainToolCall]]


def is_langchain_tool_call(obj: Any) -> bool:
    """Check if an object is a LangChain tool call (request)."""
    return (
        isinstance(obj, dict)
        and "name" in obj
        and "args" in obj
        and isinstance(obj.get("args"), dict)
    )


def is_tool_execution_result(obj: Any) -> bool:
    """Check if an object is our ToolExecutionResult (completed execution)."""
    return hasattr(obj, "name") and hasattr(obj, "success") and hasattr(obj, "args")


def has_tool_calls(message: BaseMessage) -> bool:
    """
    Strongly-typed check if a message contains tool call requests.

    Only checks for the standard LangChain tool_calls attribute.
    """
    return (
        isinstance(message, ToolMessage)
        or isinstance(message, AIMessage)
        and hasattr(message, "tool_calls")
        and isinstance(message.tool_calls, list)
        and len(message.tool_calls) > 0
        and all(is_langchain_tool_call(tc) for tc in message.tool_calls)
    )


def extract_tool_call_requests(message: BaseMessage) -> List[LangChainToolCall]:
    """
    Extract tool call requests from a message with strong typing.

    Args:
        message: Any BaseMessage that might contain tool calls

    Returns:
        List of validated LangChain tool call requests
    """
    if not has_tool_calls(message):
        return []

    # Type narrowing - we know it has tool_calls at this point
    if isinstance(message, AIMessage) and message.tool_calls:
        return [
            LangChainToolCall(name=tc["name"], args=tc["args"], id=tc.get("id"))
            for tc in message.tool_calls
            if is_langchain_tool_call(tc)
        ]

    return []


def tool_call_request_to_execution_result(
    request: LangChainToolCall,
    success: bool,
    result_data: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
    execution_time_ms: Optional[float] = None,
    message_id: Optional[int] = None,
    execution_id: Optional[str] = None,
) -> ToolCall:
    """
    Convert a tool call request to an execution result.

    This bridges the gap between what the AI requested and what actually happened.
    """
    return ToolCall(
        message_id=message_id,
        name=request["name"],
        execution_id=execution_id or request.get("id"),
        success=success,
        args=request["args"],
        result_data=result_data,
        error_message=error_message,
        execution_time_ms=execution_time_ms,
    )
