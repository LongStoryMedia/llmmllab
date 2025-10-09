"""
Bidirectional message conversion utilities.
"""

from typing import List, Optional, Union
from models import Message, LangChainMessage, MessageRole
from .extraction import (
    extract_content_from_message,
    extract_content_from_langchain_message,
    _text_to_message_content_list,
)


def message_to_langchain_message(msg: Message) -> LangChainMessage:
    """Convert a Message object to a LangChainMessage object.

    IMPORTANT: Preserve tool_calls so downstream ToolExecutorNode
    can detect and execute them. Previous implementation dropped
    tool_calls resulting in zero execution even when the model
    emitted <tool_call> markup.
    """
    content_text = extract_content_from_message(msg)

    # Determine message type from role
    message_type = "human"  # Default
    if hasattr(msg, "role") and msg.role:
        role_value = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        if role_value.lower() in ("assistant", "ai", "system"):
            message_type = (
                "ai" if role_value.lower() in ("assistant", "ai") else "system"
            )

    # Debug logging for tool calls conversion
    from composer.monitoring.logging import composer_logger
    logger = composer_logger.logger.bind(component="message_conversion")
    
    logger.info(
        "Converting Message to LangChainMessage",
        has_tool_calls=hasattr(msg, 'tool_calls') and msg.tool_calls is not None,
        tool_calls_count=len(msg.tool_calls) if hasattr(msg, 'tool_calls') and msg.tool_calls else 0,
        tool_calls_preview=str(msg.tool_calls)[:200] if hasattr(msg, 'tool_calls') and msg.tool_calls else "None",
    )

    langchain_msg = LangChainMessage(
        content=content_text,
        type=message_type,
        tool_calls=msg.tool_calls,
    )
    
    logger.info(
        "Created LangChainMessage",
        lc_has_tool_calls=hasattr(langchain_msg, 'tool_calls') and langchain_msg.tool_calls is not None,
        lc_tool_calls_count=len(langchain_msg.tool_calls) if hasattr(langchain_msg, 'tool_calls') and langchain_msg.tool_calls else 0,
    )
    
    return langchain_msg


def langchain_message_to_message(
    lc_msg: LangChainMessage, conversation_id: Optional[int] = None
) -> Message:
    """
    Convert a LangChainMessage object to a Message object.

    Args:
        lc_msg: LangChainMessage object to convert
        conversation_id: Optional conversation ID for the Message

    Returns:
        Converted Message object
    """
    content_text = extract_content_from_langchain_message(lc_msg)

    # Determine role from message type
    role = MessageRole.USER  # Default
    if hasattr(lc_msg, "type") and lc_msg.type:
        msg_type = lc_msg.type.lower()
        if msg_type == "ai":
            role = MessageRole.ASSISTANT
        elif msg_type == "system":
            role = MessageRole.SYSTEM
        elif msg_type in ("user", "human"):
            role = MessageRole.USER

    return Message(
        content=_text_to_message_content_list(content_text),
        role=role,
        conversation_id=conversation_id,
        tool_calls=lc_msg.tool_calls,
    )


def convert_messages_to_langchain(messages: List[Message]) -> List[LangChainMessage]:
    """Convert a list of Message objects to LangChainMessage objects."""
    langchain_messages = []

    for msg in messages:
        if hasattr(msg, "content") and hasattr(msg, "role"):
            # Convert from Message to LangChainMessage
            langchain_messages.append(message_to_langchain_message(msg))
        else:
            # Assume already in correct format or convert to dict
            langchain_messages.append(msg)

    return langchain_messages


def convert_langchain_messages_to_messages(
    lc_messages: List[LangChainMessage], conversation_id: Optional[int] = None
) -> List[Message]:
    """
    Convert a list of LangChainMessage objects to Message objects.

    Args:
        lc_messages: List of LangChainMessage objects to convert
        conversation_id: Optional conversation ID for all Message objects

    Returns:
        List of converted Message objects
    """
    messages = []
    for lc_msg in lc_messages:
        if hasattr(lc_msg, "content") and hasattr(lc_msg, "type"):
            # Convert from LangChainMessage to Message
            messages.append(langchain_message_to_message(lc_msg, conversation_id))
        else:
            # Handle cases where the list might contain Message objects already
            if hasattr(lc_msg, "content") and hasattr(lc_msg, "role"):
                messages.append(lc_msg)
            else:
                # Try to create a Message from whatever we have
                messages.append(
                    Message(
                        content=_text_to_message_content_list(str(lc_msg)),
                        role=MessageRole.USER,
                        conversation_id=conversation_id,
                    )
                )

    return messages


def normalize_message_list(
    messages: List[Union[Message, LangChainMessage]], target_type: str = "message"
) -> List[Union[Message, LangChainMessage]]:
    """
    Normalize a list of mixed message types to a consistent format.

    Args:
        messages: List of mixed Message and LangChainMessage objects
        target_type: Target type ("message" or "langchain")

    Returns:
        List of normalized message objects
    """
    normalized = []

    if target_type == "message":
        for msg in messages:
            if isinstance(msg, LangChainMessage):
                normalized.append(langchain_message_to_message(msg))
            elif isinstance(msg, Message):
                normalized.append(msg)
            else:
                # Create Message from string or other type
                normalized.append(
                    Message(
                        content=_text_to_message_content_list(str(msg)),
                        role=MessageRole.USER,
                    )
                )
        return normalized

    else:  # target_type == "langchain"
        for msg in messages:
            if isinstance(msg, Message):
                normalized.append(message_to_langchain_message(msg))
            elif isinstance(msg, LangChainMessage):
                normalized.append(msg)
            else:
                # Create LangChainMessage from string or other type
                normalized.append(LangChainMessage(content=str(msg), type="human"))
        return normalized


def convert_message_batch(
    messages: List[Union[Message, LangChainMessage]],
    conversation_id: Optional[int] = None,
) -> tuple[List[Message], List[LangChainMessage]]:
    """
    Convert a batch of messages to both formats simultaneously.
    Useful when you need both representations.

    Args:
        messages: List of messages in either format
        conversation_id: Optional conversation ID for Message objects

    Returns:
        Tuple of (Message list, LangChainMessage list)
    """
    msg_list = normalize_message_list(messages, "message")
    lc_msg_list = normalize_message_list(messages, "langchain")

    # Ensure we have proper types
    typed_msg_list: List[Message] = [
        msg for msg in msg_list if isinstance(msg, Message)
    ]
    typed_lc_msg_list: List[LangChainMessage] = [
        msg for msg in lc_msg_list if isinstance(msg, LangChainMessage)
    ]

    # Set conversation_id on Message objects if provided
    if conversation_id:
        for msg in typed_msg_list:
            msg.conversation_id = conversation_id

    return typed_msg_list, typed_lc_msg_list
