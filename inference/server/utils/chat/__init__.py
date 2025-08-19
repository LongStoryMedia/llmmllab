"""
Chat utility functions and helpers for the chat router.
"""

from .response import create_valid_chat_response
from .message import (
    ensure_valid_message,
    ensure_message_content_list,
    extract_message_text,
    convert_string_to_message_content,
)
from .workflow import (
    should_use_agentic_workflow,
    prepare_enhanced_messages,
    format_search_query,
    parse_ddg_results,
)

__all__ = [
    "create_valid_chat_response",
    "ensure_valid_message",
    "ensure_message_content_list",
    "extract_message_text",
    "convert_string_to_message_content",
    "should_use_agentic_workflow",
    "prepare_enhanced_messages",
    "format_search_query",
    "parse_ddg_results",
]
