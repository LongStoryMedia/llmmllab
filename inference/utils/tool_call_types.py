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
    # Handle dictionary format (LangChain native)
    if isinstance(obj, dict):
        return "name" in obj and "args" in obj and isinstance(obj.get("args"), dict)

    # Handle OpenAI ChatCompletionMessageFunctionToolCall format
    if hasattr(obj, "function") and hasattr(obj, "id"):
        return hasattr(obj.function, "name") and hasattr(obj.function, "arguments")

    return False


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
    # Type narrowing - we know it has tool_calls at this point
    if isinstance(message, AIMessage) and message.tool_calls:
        requests = []
        for tc in message.tool_calls:
            if not is_langchain_tool_call(tc):
                continue

            # Handle dictionary format (LangChain native)
            if isinstance(tc, dict):
                requests.append(
                    LangChainToolCall(name=tc["name"], args=tc["args"], id=tc.get("id"))
                )
            # Handle OpenAI ChatCompletionMessageFunctionToolCall format
            elif hasattr(tc, "function") and hasattr(tc, "id"):
                import json

                try:
                    args = (
                        json.loads(tc.function.arguments)
                        if tc.function.arguments
                        else {}
                    )
                except json.JSONDecodeError:
                    args = {}

                requests.append(
                    LangChainToolCall(name=tc.function.name, args=args, id=tc.id)
                )

        return requests

    return []


def tool_call_request_to_execution_result(
    request: LangChainToolCall,
    success: Optional[bool] = None,
    result_data: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
    execution_time_ms: Optional[int] = None,
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


def extract_tool_calls_as_models(message: BaseMessage) -> List[ToolCall]:
    """
    Extract tool call requests from a LangChain message and convert to ToolCall models.

    This is a convenience function that combines extraction and conversion.

    Args:
        message: LangChain message to extract tool calls from

    Returns:
        List of ToolCall models representing the requests
    """
    langchain_requests = extract_tool_call_requests(message)
    return [
        tool_call_request_to_execution_result(
            request=request,
            success=True,  # Requests start as "successful" until execution
            execution_id=request.get("id"),
        )
        for request in langchain_requests
    ]


def has_tool_call_requests_as_models(message: BaseMessage) -> bool:
    """
    Check if a LangChain message has tool call requests (convenience wrapper).

    Args:
        message: LangChain message to check

    Returns:
        True if message has tool call requests, False otherwise
    """
    return has_tool_calls(message)
