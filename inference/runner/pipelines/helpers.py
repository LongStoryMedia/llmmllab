"""
Helper functions for message processing.
"""

import datetime
import json
from typing import Any
import logging
import uuid

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from models import (
    MessageContent,
    MessageContentType,
    MessageRole,
    Message,
    ChatResponse,
)

logger = logging.getLogger(__name__)


def get_role(role: MessageRole) -> str:
    """
    Convert protobuf MessageRole to string representation.
    """
    if role == MessageRole.USER:
        return "user"
    elif role == MessageRole.ASSISTANT:
        return "assistant"
    elif role == MessageRole.SYSTEM:
        return "system"
    elif role == MessageRole.AGENT:
        return "agent"
    elif role == MessageRole.TOOL:
        return "tool"
    elif role == MessageRole.OBSERVER:
        return "observer"
    else:
        raise ValueError(f"Unknown message role: {role}")


def to_lc_message(message: Message) -> BaseMessage:
    """Convert a Message object to a format suitable for LangChain."""
    # Extract text content as a simple string
    text_content = extract_message_text(message)

    if message.role == MessageRole.ASSISTANT:
        return AIMessage(content=text_content)
    elif message.role == MessageRole.USER:
        return HumanMessage(content=text_content)
    elif message.role == MessageRole.SYSTEM:
        return SystemMessage(content=text_content)
    else:
        # Default to HumanMessage for unknown roles to ensure we always return a BaseMessage
        logger.warning(
            f"Unknown message role: {message.role}, defaulting to HumanMessage"
        )
        return HumanMessage(content=text_content)


def create_streaming_chunk(
    text: str, done: bool = False, role: MessageRole = MessageRole.ASSISTANT
) -> ChatResponse:
    """Create streaming chunk (preserved from legacy)."""
    message = None
    if text or not done:
        message = Message(
            role=role,
            content=(
                [MessageContent(type=MessageContentType.TEXT, text=text)]
                if text
                else []
            ),
        )

    return ChatResponse(
        done=done,
        message=message,
        created_at=datetime.datetime.now(datetime.timezone.utc),
        finish_reason="stop" if done else None,
    )


def create_error_response(error_message: str) -> ChatResponse:
    """Create standardized error response (preserved from legacy)."""
    return ChatResponse(
        done=True,
        message=Message(
            role=MessageRole.ASSISTANT,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text=f"I apologize, but I encountered an error: {error_message}",
                )
            ],
        ),
        created_at=datetime.datetime.now(datetime.timezone.utc),
        finish_reason="error",
    )


def extract_message_text(message: Message) -> str:
    """Extract text content from a message object"""
    text_parts = []
    for content in message.content:
        if content.type == MessageContentType.TEXT and content.text:
            text_parts.append(content.text)
    return "\n".join(text_parts).strip()


def serialize_to_json(obj: Any) -> str:
    """
    Serialize an object to JSON with enhanced object handling.
    Converts complex Python objects to JSON-serializable formats.

    Args:
        obj: The object to serialize (dict, list, custom object, etc.)

    Returns:
        JSON string representation of the object
    """
    return json.dumps(obj, default=_json_serializer)


def _json_serializer(obj: Any) -> Any:
    """
    Custom serializer for JSON encoding.
    Handles various types of objects that aren't JSON serializable by default.

    Args:
        obj: The object to serialize

    Returns:
        JSON-serializable representation of the object (str, dict, etc.)

    Handles:
    - Basic JSON types (str, int, float, bool, list, dict) - passed through
    - UUID objects (converts to string)
    - Date/Time objects with isoformat method (converts to ISO format string)
    - Objects with __dict__ attribute (converts to dictionary)
    """
    # Handle basic JSON serializable types
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (list, tuple, set)):
        return list(obj)
    elif isinstance(obj, dict):
        return {str(k): v for k, v in obj.items()}  # Ensure keys are strings
    elif isinstance(obj, uuid.UUID):
        return str(obj)
    elif hasattr(obj, "isoformat") and callable(getattr(obj, "isoformat")):
        # Handle datetime, date, time objects
        return obj.isoformat()
    elif hasattr(obj, "__dict__"):
        # Try to convert object to a dict of its attributes
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}

    # If we can't handle it, let the default JSON serializer raise the error
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
