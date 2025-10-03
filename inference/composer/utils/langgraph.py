"""
Helpers for building LangGraph-compatible state from various message types.
This keeps custom logic out of schema-generated models and centralizes
message conversion logic for the composer module.
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Union
import asyncio

from models import Message, LangChainMessage, UserConfig
from models.lang_graph_state import LangGraphState
from composer.graph.state import WorkflowState

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


def extract_content_from_message(msg: Message) -> str:
    """Extract text content from a Message object, handling MessageContent lists."""
    if not hasattr(msg, "content"):
        return str(msg) if msg else ""

    content = msg.content

    # Handle list of MessageContent objects
    if isinstance(content, list):
        content_parts = []
        for content_part in content:
            if hasattr(content_part, "text"):
                content_parts.append(content_part.text)
            elif isinstance(content_part, str):
                content_parts.append(content_part)
            elif hasattr(content_part, "content"):
                content_parts.append(str(content_part.content))
            else:
                content_parts.append(str(content_part))
        return "\n".join(content_parts)

    # Handle single content
    return str(content) if content else ""


def message_to_langchain_message(msg: Message) -> LangChainMessage:
    """Convert a Message object to a LangChainMessage object."""
    content_text = extract_content_from_message(msg)

    # Determine message type from role
    message_type = "human"  # Default
    if hasattr(msg, "role") and msg.role:
        role_value = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        if role_value.lower() in ("assistant", "ai", "system"):
            message_type = (
                "ai" if role_value.lower() in ("assistant", "ai") else "system"
            )

    return LangChainMessage(
        content=content_text,
        type=message_type,
    )


def convert_messages_to_langchain(messages: List[Message]) -> List[LangChainMessage]:
    """Convert a list of Message objects to LangChainMessage objects."""
    langchain_messages = []

    for msg in messages:
        if hasattr(msg, "content") and hasattr(msg, "role"):
            # Convert from Message to LangChainMessage
            langchain_messages.append(message_to_langchain_message(msg))
        else:
            # Assume already in correct format or convert to dict
            langchain_messages.append(msg)

    return langchain_messages


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
    if isinstance(item, LCBaseMessage):
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


def build_lc_messages(messages: Iterable[Any]) -> list:
    """Build a list of LangChain BaseMessage from heterogeneous items."""
    return [coerce_to_lc_message(m) for m in (messages or [])]


def build_workflow_state(
    user_id: str,
    messages: List[Message],
    user_config: UserConfig,
    additional_context: Optional[Dict[str, Any]] = None,
) -> WorkflowState:
    """Create a WorkflowState from Message objects with proper conversion."""
    # Convert messages using the centralized conversion logic
    langchain_messages = convert_messages_to_langchain(messages)

    # Create the state
    state = WorkflowState(
        messages=langchain_messages,
        user_id=user_id,
        execution_metadata={
            "created_at": asyncio.get_event_loop().time(),
            "composer_version": "0.1.0",
            # Include user workflow preferences in metadata
            "streaming_enabled": (
                getattr(user_config.workflow, "enable_streaming", True)
                if hasattr(user_config, "workflow")
                else True
            ),
            "workflow_timeout": (
                getattr(user_config.workflow, "default_timeout", 300)
                if hasattr(user_config, "workflow")
                else 300
            ),
        },
    )

    # Add additional context
    if additional_context:
        for key, value in additional_context.items():
            state.execution_metadata[key] = value

    return state


def build_langgraph_state(
    messages: Iterable[Any],
    user_input: str,
    *,
    error_count: int = 0,
    max_iterations: int = 10,
    current_iteration: int = 0,
    tools_used: Iterable[str] | None = None,
    intermediate_results: Dict[str, Any] | None = None,
) -> LangGraphState:
    """Construct a LangGraphState from heterogeneous message inputs safely."""
    msg_list = list(messages) if messages is not None else []
    coerced = [_coerce_to_langchain_message_dict(m) for m in msg_list]

    return LangGraphState(
        messages=coerced,  # type: ignore[arg-type]
        user_input=user_input or "",
        error_count=error_count,
        max_iterations=max_iterations,
        current_iteration=current_iteration,
        tools_used=list(tools_used or []),
        intermediate_results=dict(intermediate_results or {}),
    )
