"""
Helpers for building LangGraph-compatible state from various message types.
This keeps custom logic out of schema-generated models.
"""

from __future__ import annotations
from typing import Any, Dict, Iterable

from models.lang_graph_state import LangGraphState

try:
    # LangChain message classes for reconstruction
    from langchain_core.messages import (
        BaseMessage as LCBaseMessage,
        AIMessage as LCAIMessage,
        HumanMessage as LCHumanMessage,
        SystemMessage as LCSystemMessage,
    )
except Exception:  # pragma: no cover
    LCBaseMessage = object  # type: ignore
    LCAIMessage = None  # type: ignore
    LCHumanMessage = None  # type: ignore
    LCSystemMessage = None  # type: ignore


def _coerce_to_langchain_message_dict(item: Any) -> Dict[str, Any]:
    """Coerce an arbitrary object (LangChain BaseMessage or dict/str) into
    a dict matching the LangChainMessage schema used by LangGraphState.
    """
    # If it's already a dict, assume it's compliant
    if isinstance(item, dict):
        return item

    # Duck-typing for LangChain BaseMessage-like objects
    if hasattr(item, "content"):
        return {
            "content": getattr(item, "content", ""),
            "additional_kwargs": getattr(item, "additional_kwargs", {}) or {},
            "response_metadata": getattr(item, "response_metadata", {}) or {},
            "type": getattr(item, "type", "text") or "text",
            "name": getattr(item, "name", None),
            "id": getattr(item, "id", None),
        }

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
    # Already a BaseMessage
    if isinstance(item, LCBaseMessage):
        return item

    # Expect dict-like
    if isinstance(item, dict):
        content = item.get("content", "")
        mtype = (item.get("type") or item.get("role") or "").lower()
        if mtype in ("ai", "assistant") and LCAIMessage is not None:
            return LCAIMessage(content=content)
        if mtype in ("human", "user") and LCHumanMessage is not None:
            return LCHumanMessage(content=content)
        if mtype == "system" and LCSystemMessage is not None:
            return LCSystemMessage(content=content)
        # Default fallback
        if LCHumanMessage is not None:
            return LCHumanMessage(content=content)
        return item  # last resort

    # Fallback: string to HumanMessage
    if LCHumanMessage is not None:
        return LCHumanMessage(content=str(item))
    return item


def build_lc_messages(messages: Iterable[Any]) -> list:
    """Build a list of LangChain BaseMessage from heterogeneous items."""
    return [coerce_to_lc_message(m) for m in (messages or [])]


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
