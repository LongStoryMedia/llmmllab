"""
Utility functions for handling message processing in chat endpoints.
"""

from typing import List, Dict, Any, Optional
from models.message import Message
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from models.message_role import MessageRole
from datetime import datetime as dt


def extract_message_text(message: Message) -> str:
    """Extract text content from a message object"""
    text_parts = []
    for content in message.content:
        if content.type == MessageContentType.TEXT and content.text:
            text_parts.append(content.text)
    return " ".join(text_parts).strip()


def convert_string_to_message_content(message_text: str) -> List[MessageContent]:
    """Convert a plain string to a list of MessageContent objects"""
    return [MessageContent(type=MessageContentType.TEXT, text=message_text, url=None)]


def ensure_valid_message(message, default_conversation_id: int = -1):
    """
    Ensure a message object is valid by checking and fixing both content and conversation_id.

    Args:
        message: The message object to validate
        default_conversation_id: The default conversation ID to use if none is provided

    Returns:
        A valid message object with proper content format and conversation_id
    """
    # If message is a dict, convert to Message object
    if isinstance(message, dict):
        # Extract values from dict
        role = message.get("role", MessageRole.USER)
        content = message.get("content", "")

        # Ensure conversation_id is set - this is critical
        conversation_id = message.get("conversation_id")
        if conversation_id is None:
            conversation_id = message.get("id_conversation")  # Try alternate format
        if conversation_id is None:
            conversation_id = default_conversation_id  # Fall back to default

        message_id = message.get("message_id", message.get("id"))
        created_at = message.get("created_at", message.get("timestamp", dt.now()))
        thinking = message.get("thinking")
        tool_calls = message.get("tool_calls")

        # Format content as MessageContent list
        if isinstance(content, str):
            content = convert_string_to_message_content(content)
        elif isinstance(content, list) and all(
            isinstance(item, MessageContent) for item in content
        ):
            # Content is already in correct format
            pass
        elif isinstance(content, list):
            # List of something else - convert each item
            formatted_content = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    formatted_content.append(
                        MessageContent(
                            type=item.get("type", MessageContentType.TEXT),
                            text=item["text"],
                            url=item.get("url"),
                        )
                    )
                else:
                    text = str(item) if item is not None else ""
                    formatted_content.append(
                        MessageContent(
                            type=MessageContentType.TEXT, text=text, url=None
                        )
                    )
            content = formatted_content
        else:
            # Some other format - convert to string
            try:
                text = str(content) if content is not None else ""
                content = convert_string_to_message_content(text)
            except:
                content = [
                    MessageContent(type=MessageContentType.TEXT, text="", url=None)
                ]

        # Ensure we have a valid conversation_id before creating the Message
        if conversation_id is None or not isinstance(conversation_id, int):
            conversation_id = default_conversation_id

        # Create new Message object with required fields
        try:
            message = Message(
                role=role,
                content=content,
                conversation_id=conversation_id,
                id=message_id,
                created_at=created_at,
                thinking=thinking,
                tool_calls=tool_calls,
            )
        except Exception as e:
            # If creation fails, try with minimal required fields
            message = Message(
                role=MessageRole.USER if not isinstance(role, MessageRole) else role,
                content=[
                    MessageContent(type=MessageContentType.TEXT, text="", url=None)
                ],
                conversation_id=default_conversation_id,
            )

    # Now message should be a Message object
    # Ensure content is a list of MessageContent objects
    try:
        if message.content is None:
            # Handle None content
            message.content = [
                MessageContent(type=MessageContentType.TEXT, text="", url=None)
            ]
        elif not isinstance(message.content, list):
            # If content is a string, convert it to a list with one MessageContent
            if isinstance(message.content, str):
                message.content = convert_string_to_message_content(message.content)
            else:
                # Handle other types
                try:
                    text = str(message.content) if message.content is not None else ""
                    message.content = convert_string_to_message_content(text)
                except Exception:
                    message.content = [
                        MessageContent(type=MessageContentType.TEXT, text="", url=None)
                    ]
    except Exception:
        # Fallback for any unexpected errors
        message.content = [
            MessageContent(type=MessageContentType.TEXT, text="", url=None)
        ]

    # Ensure each item in the list is a MessageContent
    for i, content in enumerate(message.content):
        if not isinstance(content, MessageContent):
            try:
                # Try to convert to MessageContent
                text = (
                    content.get("text", "")
                    if isinstance(content, dict)
                    else str(content)
                )
                message.content[i] = MessageContent(
                    type=MessageContentType.TEXT, text=text, url=None
                )
            except:
                message.content[i] = MessageContent(
                    type=MessageContentType.TEXT, text="", url=None
                )

    # Ensure conversation_id exists and is valid
    if not hasattr(message, "conversation_id"):
        message.conversation_id = default_conversation_id
    elif message.conversation_id is None:
        message.conversation_id = default_conversation_id
    elif not isinstance(message.conversation_id, int):
        try:
            # Try to convert to int
            message.conversation_id = int(message.conversation_id)
        except:
            message.conversation_id = default_conversation_id

    # Final check for conversation_id
    try:
        if not message.conversation_id and message.conversation_id != 0:
            message.conversation_id = (
                default_conversation_id if default_conversation_id != 0 else -1
            )
    except Exception:
        message.conversation_id = (
            default_conversation_id if default_conversation_id != 0 else -1
        )

    return message


def ensure_message_content_list(message):
    """Ensure message content is a list of MessageContent objects"""
    if not isinstance(message.content, list):
        # If content is a string, convert it to a list with one MessageContent
        if isinstance(message.content, str):
            message.content = convert_string_to_message_content(message.content)
        else:
            # Handle other types (like dict)
            try:
                text = str(message.content)
                message.content = convert_string_to_message_content(text)
            except:
                message.content = [
                    MessageContent(type=MessageContentType.TEXT, text="", url=None)
                ]

    # Ensure each item in the list is a MessageContent
    for i, content in enumerate(message.content):
        if not isinstance(content, MessageContent):
            try:
                # Try to convert to MessageContent
                text = (
                    content.get("text", "")
                    if isinstance(content, dict)
                    else str(content)
                )
                message.content[i] = MessageContent(
                    type=MessageContentType.TEXT, text=text, url=None
                )
            except:
                message.content[i] = MessageContent(
                    type=MessageContentType.TEXT, text="", url=None
                )

    return message
