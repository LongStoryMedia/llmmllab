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

__all__ = [
    "get_model_profile_for_task",
    "get_profile_id_for_task",
    "get_model_profile",
    "extract_message_text",
    "ensure_message_content_list",
    "get_grammar_for_model",
    "parse_structured_output",
    "StructuredOutputError",
]
