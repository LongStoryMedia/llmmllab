"""
Unified message conversion utilities for converting between internal Message objects and LangChain BaseMessage types.

This module consolidates all message conversion logic to eliminate duplicate implementations
and provide a single source of truth for message format conversion.
"""

from typing import List, Optional, Union, Dict, Any
from datetime import datetime, timezone

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)

from models import (
    Message,
    MessageRole,
    MessageContent,
    MessageContentType,
)
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="message_conversion")


def message_to_base_message(message: Message) -> BaseMessage:
    """Convert a Message object to a LangChain BaseMessage."""
    # Extract text content as a simple string
    text_content = extract_text_from_message(message)

    if message.role == MessageRole.ASSISTANT:
        return AIMessage(content=text_content)
    elif message.role == MessageRole.USER:
        return HumanMessage(content=text_content)
    elif message.role == MessageRole.SYSTEM:
        return SystemMessage(content=text_content)
    else:
        # Default to human message for unknown roles
        return HumanMessage(content=text_content)


def base_message_to_message(
    base_message: BaseMessage, conversation_id: Optional[int] = None
) -> Message:
    """Convert a LangChain BaseMessage to a Message object."""

    # Determine role based on message type
    if isinstance(base_message, AIMessage):
        role = MessageRole.ASSISTANT
        message_type = "ai"
    elif isinstance(base_message, HumanMessage):
        role = MessageRole.USER
        message_type = "human"
    elif isinstance(base_message, SystemMessage):
        role = MessageRole.SYSTEM
        message_type = "system"
    elif isinstance(base_message, ToolMessage):
        role = MessageRole.ASSISTANT  # Tool messages are typically assistant responses
        message_type = "tool"
    else:
        # Default to user role for unknown types
        role = MessageRole.USER
        message_type = "human"

    # Handle content - preserve multimodal structure or convert to MessageContent
    content = []
    if isinstance(base_message.content, list):
        # Multimodal content - convert each item to MessageContent
        for item in base_message.content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    content.append(
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=item.get("text", ""),
                            url=None,
                        )
                    )
                elif item.get("type") == "image_url":
                    content.append(
                        MessageContent(
                            type=MessageContentType.IMAGE,
                            text=None,
                            url=item.get("image_url", {}).get("url", ""),
                        )
                    )
                else:
                    # Unknown content type, treat as text
                    content.append(
                        MessageContent(
                            type=MessageContentType.TEXT, text=str(item), url=None
                        )
                    )
            else:
                # String content item
                content.append(
                    MessageContent(
                        type=MessageContentType.TEXT, text=str(item), url=None
                    )
                )
    else:
        # Simple string content
        content = [
            MessageContent(
                type=MessageContentType.TEXT,
                text=str(base_message.content) if base_message.content else "",
                url=None,
            )
        ]

    return Message(
        role=role,
        content=content,
        conversation_id=conversation_id,
        created_at=datetime.now(timezone.utc),
    )


def messages_to_base_messages(messages: List[Message]) -> List[BaseMessage]:
    """Convert a list of Message objects to LangChain BaseMessages."""
    return [message_to_base_message(msg) for msg in messages]


def base_messages_to_messages(
    base_messages: List[BaseMessage], conversation_id: Optional[int] = None
) -> List[Message]:
    """Convert a list of LangChain BaseMessages to Message objects."""
    return [base_message_to_message(msg, conversation_id) for msg in base_messages]


def extract_text_from_message(message: Message) -> str:
    """Extract text content from a Message object."""
    if not message.content:
        return ""

    text_parts = []
    for content in message.content:
        if isinstance(content, MessageContent):
            if content.type == MessageContentType.TEXT and content.text:
                text_parts.append(content.text)
        elif isinstance(content, dict):
            # Handle dictionary format
            if content.get("type") == "text" and content.get("text"):
                text_parts.append(content["text"])
        else:
            # Fallback for other types
            text_str = str(content)
            if text_str:
                text_parts.append(text_str)

    return "\n".join(text_parts) if text_parts else ""


def extract_text_from_base_message(base_message: BaseMessage) -> str:
    """Extract text content from a LangChain BaseMessage."""
    if not hasattr(base_message, "content"):
        return ""

    content = base_message.content

    # Handle multimodal content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        return "\n".join(text_parts)

    # Handle simple string content
    return str(content) if content else ""


def get_most_recent_user_message_text(messages: List[BaseMessage]) -> str:
    """Extract text from the most recent user message in a conversation."""
    if not messages:
        return ""

    # Look for the most recent user message
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return extract_text_from_base_message(msg)

    # Fallback: if no user message found, use the last message
    if messages:
        return extract_text_from_base_message(messages[-1])

    return ""


def create_text_message_content(text: str) -> List[MessageContent]:
    """Create a list containing a single MessageContent object with text content."""
    return [MessageContent(type=MessageContentType.TEXT, text=text, url=None)]


def normalize_message_input(
    input_data: Union[
        str, Message, List[Union[str, Message]], List[str], List[Message]
    ],
    role: MessageRole = MessageRole.USER,
) -> List[Message]:
    """
    Normalize various message input formats to a list of Message objects.

    Args:
        input_data: String, Message, or list of strings/messages
        role: Default role for string inputs

    Returns:
        List of normalized Message objects
    """
    if isinstance(input_data, str):
        return [
            Message(
                role=role,
                content=create_text_message_content(input_data),
                created_at=datetime.now(timezone.utc),
            )
        ]
    elif isinstance(input_data, Message):
        return [input_data]
    elif isinstance(input_data, list):
        normalized = []
        for item in input_data:
            if isinstance(item, str):
                normalized.append(
                    Message(
                        role=role,
                        content=create_text_message_content(item),
                        created_at=datetime.now(timezone.utc),
                    )
                )
            elif isinstance(item, Message):
                normalized.append(item)
            else:
                # Handle other types by converting to string
                normalized.append(
                    Message(
                        role=role,
                        content=create_text_message_content(str(item)),
                        created_at=datetime.now(timezone.utc),
                    )
                )
        return normalized
    else:
        # Fallback: convert to string and create message
        return [
            Message(
                role=role,
                content=create_text_message_content(str(input_data)),
                created_at=datetime.now(timezone.utc),
            )
        ]
