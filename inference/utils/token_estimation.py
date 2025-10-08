"""
Token estimation utilities for consistent token counting across the system.
Provides centralized token estimation logic to avoid duplication.
"""

from typing import Any, List, Dict, Union
from models import Memory
from models.lang_chain_message import LangChainMessage


def estimate_tokens(content: Any) -> int:
    """
    Estimate token count for any content type using consistent approximation.

    Uses the standard approximation of ~4 characters per token, which works
    reasonably well for English text and code.

    Args:
        content: Content to estimate tokens for (str, list, dict, object)

    Returns:
        Approximate token count
    """
    if isinstance(content, str):
        # Basic approximation: ~4 characters per token
        return len(content) // 4
    elif isinstance(content, list):
        return sum(estimate_tokens(item) for item in content)
    elif isinstance(content, dict):
        # Estimate tokens for all values in the dictionary
        return sum(estimate_tokens(value) for value in content.values())
    elif hasattr(content, "__dict__"):
        # For objects, convert to string representation
        return estimate_tokens(str(content))
    else:
        # Fallback: convert to string and estimate
        return len(str(content)) // 4


def estimate_message_tokens(message: LangChainMessage) -> int:
    """
    Calculate approximate token count for a LangChain message.

    Includes content, role formatting, and metadata overhead.

    Args:
        message: LangChain message to calculate tokens for

    Returns:
        Approximate token count including formatting overhead
    """
    content_tokens = estimate_tokens(message.content)
    role_tokens = 5  # For role formatting (User:, Assistant:, etc.)
    metadata_tokens = 10  # For metadata overhead (timestamps, etc.)

    return content_tokens + role_tokens + metadata_tokens


def estimate_memory_tokens(memories: List[Memory]) -> int:
    """
    Calculate approximate token count for a list of memories.

    Includes content from all fragments plus formatting overhead.

    Args:
        memories: List of Memory objects

    Returns:
        Approximate token count including formatting
    """
    total_tokens = 0

    for memory in memories:
        for fragment in memory.fragments:
            # Content tokens
            total_tokens += estimate_tokens(fragment.content)

            # Formatting overhead (role labels, similarity scores, etc.)
            total_tokens += 15  # Slightly higher overhead for memory formatting

    return total_tokens


def estimate_text_with_overhead(text: str, overhead_tokens: int = 10) -> int:
    """
    Estimate tokens for text content with additional formatting overhead.

    Useful for text that will be formatted with labels, timestamps, etc.

    Args:
        text: Text content to estimate
        overhead_tokens: Additional tokens for formatting overhead

    Returns:
        Token count including overhead
    """
    return estimate_tokens(text) + overhead_tokens


def estimate_structured_content_tokens(
    content: Dict[str, Any], include_keys: bool = True
) -> int:
    """
    Estimate tokens for structured content like JSON or dictionaries.

    Args:
        content: Dictionary or structured content
        include_keys: Whether to include dictionary keys in token count

    Returns:
        Approximate token count
    """
    total_tokens = 0

    if isinstance(content, dict):
        for key, value in content.items():
            if include_keys:
                total_tokens += estimate_tokens(key)
            total_tokens += estimate_tokens(value)
            # Add small overhead for JSON formatting (colons, commas, etc.)
            total_tokens += 2
    else:
        total_tokens = estimate_tokens(content)

    return total_tokens


def get_token_budget_info(
    estimated_tokens: int, max_tokens: int
) -> Dict[str, Union[int, float, bool]]:
    """
    Get token budget information for logging and optimization.

    Args:
        estimated_tokens: Current estimated token usage
        max_tokens: Maximum token budget

    Returns:
        Dictionary with budget information
    """
    percentage_used = (estimated_tokens / max_tokens) * 100 if max_tokens > 0 else 100
    tokens_remaining = max_tokens - estimated_tokens

    return {
        "estimated_tokens": estimated_tokens,
        "max_tokens": max_tokens,
        "tokens_remaining": tokens_remaining,
        "percentage_used": round(percentage_used, 2),
        "over_budget": estimated_tokens > max_tokens,
        "near_limit": percentage_used > 90,
    }


# Legacy compatibility functions for existing code
def calculate_memory_token_count(memories: List[Memory]) -> int:
    """Legacy compatibility function - use estimate_memory_tokens instead."""
    return estimate_memory_tokens(memories)


def calculate_message_tokens(message: LangChainMessage) -> int:
    """Legacy compatibility function - use estimate_message_tokens instead."""
    return estimate_message_tokens(message)
