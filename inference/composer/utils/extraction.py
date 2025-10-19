"""
Message content extraction utilities.

This module provides functions for extracting text content from various
message types including Message and LangChainMessage objects.
"""

from typing import List

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from models import Message, LangChainMessage
from models.message_content import MessageContent
from models.message_content_type import MessageContentType


def _create_text_message_content(text: str) -> MessageContent:
    """Create a MessageContent object with text content."""
    return MessageContent(type=MessageContentType.TEXT, text=text, url=None)


def _text_to_message_content_list(text: str) -> List[MessageContent]:
    """Convert a text string to a list containing a single MessageContent object."""
    return [_create_text_message_content(text)]


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
            else:
                content_parts.append(str(content_part))
        return "\n".join(content_parts)

    # Handle single content
    return str(content) if content else ""


def extract_content_from_langchain_message(msg: LangChainMessage) -> str:
    """Extract text content from a LangChainMessage object."""
    if not hasattr(msg, "content"):
        return str(msg) if msg else ""

    content = msg.content

    # Handle list content (LangChainMessage supports string or object items)
    if isinstance(content, list):
        content_parts = []
        for part in content:
            if isinstance(part, str):
                content_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                content_parts.append(part["text"])
            else:
                # Handle any other object
                content_parts.append(str(part))
        return " ".join(content_parts)

    # Handle single content
    return str(content) if content else ""


def extract_content_from_base_langchain_message(msg: BaseMessage) -> str:
    """Extract text content from a BaseLangChainMessage object."""
    if not hasattr(msg, "content"):
        return str(msg) if msg else ""

    content = msg.content

    # Handle list content (BaseLangChainMessage supports string or object items)
    if isinstance(content, list):
        content_parts = []
        for part in content:
            if isinstance(part, str):
                content_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                content_parts.append(part["text"])
            else:
                # Handle any other object
                content_parts.append(str(part))
        return " ".join(content_parts)

    # Handle single content
    return str(content) if content else ""


def get_most_recent_user_message_content(messages: List[LangChainMessage]) -> str:
    """
    Extract content from the most recent user message in a conversation.

    Args:
        messages: List of LangChainMessage objects

    Returns:
        Content of the most recent user message, or empty string if none found
    """
    if not messages:
        return ""

    # Look for the most recent user message by checking message type
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type in ("user", "human"):
            return extract_content_from_langchain_message(msg)

    # Fallback: if no explicit user message found, use the last message
    # This handles cases where message types might not be set properly
    if messages:
        return extract_content_from_langchain_message(messages[-1])

    return ""
