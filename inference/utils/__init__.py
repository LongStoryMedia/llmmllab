"""
Shared utility modules for the inference system.
"""

from .message_conversion import (
    extract_text_from_message,
    create_text_message_content,
    message_to_lc_message,
    lc_message_to_message,
    messages_to_lc_messages,
    lc_messages_to_messages,
    convert_message_content_to_langchain_format,
    convert_lc_message_content_to_message_format,
    get_most_recent_user_message_text,
    normalize_message_input,
)
from .tool_call_types import (
    extract_tool_calls_as_models,
    has_tool_call_requests_as_models,
    extract_tool_call_requests,
    tool_call_request_to_execution_result,
    has_tool_calls,
    is_langchain_tool_call,
)
from .tool_call_extraction import (
    extract_tool_calls_from_langchain_message,
    has_tool_calls_in_langchain_message,
    extract_tool_calls_from_streaming_chunks,
    extract_tool_calls_from_message_content,
    create_tool_call_message_content,
)

from .grammar_generator import (
    get_grammar_for_model,
    parse_structured_output,
    StructuredOutputError,
)

from .token_estimation import (
    estimate_tokens,
    estimate_message_tokens,
    estimate_memory_tokens,
    estimate_text_with_overhead,
    estimate_structured_content_tokens,
    get_token_budget_info,
    calculate_memory_token_count,
    calculate_message_tokens,
)

from .response import create_streaming_chunk, create_error_response

__all__ = [
    "get_grammar_for_model",
    "parse_structured_output",
    "StructuredOutputError",
    "estimate_tokens",
    "estimate_message_tokens",
    "estimate_memory_tokens",
    "estimate_text_with_overhead",
    "estimate_structured_content_tokens",
    "get_token_budget_info",
    "calculate_memory_token_count",
    "calculate_message_tokens",
    "create_streaming_chunk",
    "create_error_response",
    # Message conversion functions
    "extract_text_from_message",
    "create_text_message_content",
    "message_to_lc_message",
    "lc_message_to_message",
    "messages_to_lc_messages",
    "lc_messages_to_messages",
    "convert_message_content_to_langchain_format",
    "convert_lc_message_content_to_message_format",
    "get_most_recent_user_message_text",
    "normalize_message_input",
    # Tool call functions
    "extract_tool_calls_as_models",
    "has_tool_call_requests_as_models",
    "extract_tool_call_requests",
    "tool_call_request_to_execution_result",
    "has_tool_calls",
    "is_langchain_tool_call",
    "extract_tool_calls_from_langchain_message",
    "has_tool_calls_in_langchain_message",
    "extract_tool_calls_from_streaming_chunks",
    "extract_tool_calls_from_message_content",
    "create_tool_call_message_content",
]
