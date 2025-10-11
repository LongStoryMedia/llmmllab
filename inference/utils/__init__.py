"""
Shared utility modules for the inference system.
"""

from .model_profile import (
    get_model_profile_for_task,
    get_profile_id_for_task,
    get_model_profile,
)

from .message import extract_message_text, ensure_message_content_list

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
    "get_model_profile_for_task",
    "get_profile_id_for_task",
    "get_model_profile",
    "extract_message_text",
    "ensure_message_content_list",
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
]
