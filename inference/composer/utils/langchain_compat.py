"""
LangChain compatibility utilities for message conversion and interoperability.

This module handles the complexities of converting between our Message models
and LangChain's BaseMessage types, including tool calls and metadata preservation.
"""

from typing import Any, Dict, Iterable, List
from .conversion import message_to_langchain_message

# LangChain message classes for reconstruction
try:
    from langchain_core.messages import (
        BaseMessage as LCBaseMessage,
        AIMessage as LCAIMessage,
        HumanMessage as LCHumanMessage,
        SystemMessage as LCSystemMessage,
        ToolMessage as LCToolMessage,
    )
except ImportError:
    # Fallback if langchain_core is not available
    LCBaseMessage = None
    LCAIMessage = None
    LCHumanMessage = None
    LCSystemMessage = None
    LCToolMessage = None


def _coerce_to_langchain_message_dict(item: Any) -> Dict[str, Any]:
    """Coerce an arbitrary object (LangChain BaseMessage or dict/str) into
    a dict matching the LangChainMessage schema used by LangGraphState.
    """
    # Handle Message objects specifically
    if hasattr(item, "content") and hasattr(item, "role"):
        lc_msg = message_to_langchain_message(item)
        return {
            "content": lc_msg.content,
            "type": lc_msg.type,
            "additional_kwargs": {},
            "response_metadata": {},
            "name": None,
            "id": None,
        }

    # If it's already a dict, assume it's compliant
    if isinstance(item, dict):
        return item

    # Special handling for ToolMessage
    if LCToolMessage and isinstance(item, LCToolMessage):
        return {
            "content": getattr(item, "content", ""),
            "additional_kwargs": getattr(item, "additional_kwargs", {}) or {},
            "response_metadata": getattr(item, "response_metadata", {}) or {},
            "type": "tool",
            "name": getattr(item, "name", None),
            "id": getattr(item, "id", None),
            "tool_call_id": getattr(item, "tool_call_id", None),
        }

    # Duck-typing for LangChain BaseMessage-like objects
    if hasattr(item, "content"):
        result = {
            "content": getattr(item, "content", ""),
            "additional_kwargs": getattr(item, "additional_kwargs", {}) or {},
            "response_metadata": getattr(item, "response_metadata", {}) or {},
            "type": getattr(item, "type", "text") or "text",
            "name": getattr(item, "name", None),
            "id": getattr(item, "id", None),
        }

        # Preserve tool_calls if present (important for LangGraph tool routing)
        if hasattr(item, "tool_calls") and getattr(item, "tool_calls", None):
            result["tool_calls"] = getattr(item, "tool_calls")

        # Preserve tool_call_id for ToolMessage-like objects
        if hasattr(item, "tool_call_id"):
            result["tool_call_id"] = getattr(item, "tool_call_id", None)

        return result

    # Fallback: stringify
    return {
        "content": str(item) if item is not None else "",
        "additional_kwargs": {},
        "response_metadata": {},
        "type": "text",
        "name": None,
        "id": None,
    }


def coerce_to_langchain_message_dict(item: Any) -> Dict[str, Any]:
    """Public helper: coerce a message-like object to the LangChainMessage dict."""
    return _coerce_to_langchain_message_dict(item)


def coerce_to_lc_message(item: Any) -> Any:
    """Convert dict/schema message into a LangChain BaseMessage for LLMs.
    Falls back to HumanMessage for unknown types.
    """
    if item is None:
        return item

    # Already a BaseMessage
    if LCBaseMessage is not None and isinstance(item, LCBaseMessage):
        return item

    # Handle pydantic models or dict-like objects
    content = ""
    mtype = ""
    tool_calls = None

    if isinstance(item, dict):
        content = item.get("content", "")
        mtype = (item.get("type") or item.get("role") or "").lower()
        tool_calls = item.get("tool_calls", None)
    elif hasattr(item, "content") and hasattr(item, "type"):
        # Pydantic model or similar with attributes
        content = getattr(item, "content", "")
        mtype = (getattr(item, "type", "") or "").lower()
        tool_calls = getattr(item, "tool_calls", None)
    else:
        # Fallback: string to HumanMessage
        if LCHumanMessage is not None:
            return LCHumanMessage(content=str(item))
        return item

    if mtype in ("ai", "assistant") and LCAIMessage is not None:
        # Preserve tool_calls for AI messages
        if tool_calls:
            return LCAIMessage(content=content, tool_calls=tool_calls)
        else:
            return LCAIMessage(content=content)
    if mtype in ("human", "user") and LCHumanMessage is not None:
        return LCHumanMessage(content=content)
    if mtype == "system" and LCSystemMessage is not None:
        return LCSystemMessage(content=content)
    if mtype == "tool" and LCToolMessage is not None:
        # Handle ToolMessage with tool_call_id
        tool_call_id = None
        if isinstance(item, dict):
            tool_call_id = item.get("tool_call_id", None)
        elif hasattr(item, "tool_call_id"):
            tool_call_id = getattr(item, "tool_call_id", None)
        return LCToolMessage(content=content, tool_call_id=tool_call_id or "unknown")
    # Default fallback
    if LCHumanMessage is not None:
        return LCHumanMessage(content=content)
    return item  # last resort


def build_lc_messages(messages: Iterable[Any]) -> List[Any]:
    """Build a list of LangChain BaseMessage from heterogeneous items."""
    return [coerce_to_lc_message(m) for m in (messages or [])]
