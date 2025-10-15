"""
Composer utilities for LangGraph integration and message handling.
"""

from .conversion import (
    message_to_langchain_message,
    langchain_message_to_message,
    convert_messages_to_langchain,
    convert_langchain_messages_to_messages,
    normalize_message_input,
)

from .extraction import (
    extract_content_from_message,
    extract_content_from_langchain_message,
    get_most_recent_user_message_content,
)

from .state import (
    build_langgraph_state,
    assemble_context_messages,
)

from .langchain_compat import (
    coerce_to_langchain_message_dict,
    coerce_to_lc_message,
    build_lc_messages,
)

__all__ = [
    # Content extraction
    "extract_content_from_message",
    "extract_content_from_langchain_message",
    "get_most_recent_user_message_content",
    # Bidirectional conversion
    "message_to_langchain_message",
    "langchain_message_to_message",
    "convert_messages_to_langchain",
    "convert_langchain_messages_to_messages",
    "normalize_message_input",
    # State building
    "build_langgraph_state",
    "assemble_context_messages",
    # LangChain compatibility
    "coerce_to_langchain_message_dict",
    "coerce_to_lc_message",
    "build_lc_messages",
]
