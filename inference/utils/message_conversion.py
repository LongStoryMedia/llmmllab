"""
Unified message conversion utilities for converting between internal Message objects and LangChain BaseMessage types.

This module consolidates all message conversion logic to eliminate duplicate implementations
and provide a single source of truth for message format conversion.
"""

import json
import re
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
from .logging import llmmllogger, serialize_event_data
from .tool_call_types import is_langchain_tool_call
from .tool_call_extraction import (
    extract_tool_calls_from_message_content,
    extract_tool_calls_from_langchain_message,
)

logger = llmmllogger.bind(component="message_conversion")

MessageInput = Union[str, Message, List[Union[str, Message]], List[str], List[Message]]


def message_to_lc_message(message: Message) -> BaseMessage:
    """Convert a Message object to a LangChain BaseMessage, preserving multimodal content."""

    # Import here to avoid circular imports

    import json

    # Convert Message.content to the multimodal format that LangChain expects
    content_data = convert_message_content_to_langchain_format(message.content)

    # For assistant messages, also parse XML tool calls from text content
    parsed_tool_calls = []
    if message.role == MessageRole.ASSISTANT or message.role == MessageRole.AGENT:
        # Extract XML-wrapped tool calls from text content
        if isinstance(content_data, str):
            # Parse <tool_call>{"name": "func", "arguments": {...}}</tool_call> format
            tool_call_pattern = (
                r"<tool_call>\s*({[^}]*(?:{[^}]*}[^}]*)*})\s*</tool_call>"
            )
            matches = re.findall(tool_call_pattern, content_data, re.DOTALL)

            for match in matches:
                try:
                    tool_call_data = json.loads(match)
                    if isinstance(tool_call_data, dict) and "name" in tool_call_data:
                        # Extract arguments - handle both dict and JSON string formats
                        args = tool_call_data.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"Failed to parse arguments string: {args}"
                                )
                                args = {}

                        # Convert to LangChain tool call format
                        lc_tool_call = {
                            "name": tool_call_data["name"],
                            "args": args,
                            "id": f"call_{tool_call_data['name']}_{len(parsed_tool_calls)}",
                        }
                        parsed_tool_calls.append(lc_tool_call)
                        logger.info(
                            f"🔧 Parsed tool call: {tool_call_data['name']} with args: {args}"
                        )
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse tool call JSON: {e}")

            # Remove tool call XML from content to clean it up
            content_data = re.sub(
                tool_call_pattern, "", content_data, flags=re.DOTALL
            ).strip()

        # Create AIMessage with parsed tool calls
        ai_message = AIMessage(content=content_data)
        if parsed_tool_calls:
            ai_message.tool_calls = parsed_tool_calls
            logger.info(
                f"🔧 AIMessage created with {len(parsed_tool_calls)} tool calls"
            )
        return ai_message

    elif message.role == MessageRole.USER:
        return HumanMessage(content=content_data)
    elif message.role == MessageRole.SYSTEM:
        return SystemMessage(content=content_data)
    elif message.role == MessageRole.TOOL:
        return ToolMessage(content=content_data)
    elif message.role == MessageRole.OBSERVER:
        # Treat observer messages as system messages
        return SystemMessage(content=content_data)
    else:
        # Default to human message for unknown roles
        return HumanMessage(content=content_data)


def lc_message_to_message(
    base_message: BaseMessage,
    conversation_id: Optional[int] = None,
) -> Message:
    """Convert a LangChain BaseMessage to a Message object."""

    # Determine role based on message type
    if isinstance(base_message, (AIMessage)):
        role = MessageRole.ASSISTANT
    elif isinstance(base_message, HumanMessage):
        role = MessageRole.USER
    elif isinstance(base_message, SystemMessage):
        role = MessageRole.SYSTEM
    elif isinstance(base_message, ToolMessage):
        role = MessageRole.TOOL  # Tool messages are typically assistant responses
    else:
        # Default to user role for unknown types
        role = MessageRole.USER

    # Handle content - preserve multimodal structure or convert to MessageContent
    content = convert_lc_message_content_to_message_format(base_message.content)
    tool_calls = extract_tool_calls_from_langchain_message(base_message)

    # Validate that content is not empty - ensure at least empty text content
    if not content:
        content = [
            MessageContent(
                type=MessageContentType.TEXT,
                text="",
                url=None,
            )
        ]

    # Create message with explicit field validation
    try:
        msg = Message(
            role=role,
            content=content,
            conversation_id=conversation_id,
            created_at=datetime.now(timezone.utc),
            tool_calls=tool_calls,
        )
    except Exception as e:
        logger.error(f"Failed to create Message object: {e}")
        logger.error(f"Role: {role}, Content: {content}")
        raise

    logger.debug(f"Converted LC message to Message: role={msg.role}, content_count={len(msg.content)}")

    return msg


def messages_to_lc_messages(messages: List[Message]) -> List[BaseMessage]:
    """Convert a list of Message objects to LangChain BaseMessages."""
    return [message_to_lc_message(msg) for msg in messages]


def lc_messages_to_messages(
    base_messages: List[BaseMessage], conversation_id: Optional[int] = None
) -> List[Message]:
    """Convert a list of LangChain BaseMessages to Message objects."""
    return [lc_message_to_message(msg, conversation_id) for msg in base_messages]


def convert_message_content_to_langchain_format(
    content: List[MessageContent],
) -> Union[str, List[Union[str, Dict[str, Any]]]]:
    """
    Convert Message.content list to LangChain multimodal format.

    Returns:
        - str: For simple text-only messages
        - List[Union[str, Dict[str, Any]]]: For multimodal messages with text and/or images
    """
    if not content:
        return ""

    # If single text content, return as string for simplicity
    if len(content) == 1 and content[0].type == MessageContentType.TEXT:
        return content[0].text or ""

    # Multimodal content - return as list of dictionaries
    result = []
    for content_item in content:
        if content_item.type == MessageContentType.TEXT:
            result.append({"type": "text", "text": content_item.text or ""})
        elif content_item.type == MessageContentType.IMAGE:
            result.append(
                {"type": "image_url", "image_url": {"url": content_item.url or ""}}
            )
        # Add other content types as needed

    return result


def convert_lc_message_content_to_message_format(
    lc_content: Union[str, List[Union[str, Dict[str, Any]]]],
) -> List[MessageContent]:
    """
    Convert LangChain BaseMessage content to Message.content format.

    Args:
        lc_content: Content from LangChain BaseMessage (str or list)

    Returns:
        - List[MessageContent]: List of MessageContent objects
    """

    content = []
    if isinstance(lc_content, list):
        # Multimodal content - convert each item to MessageContent
        for item in lc_content:
            if isinstance(item, dict):
                if is_langchain_tool_call(item.get("content", {})):
                    try:
                        content.append(
                            MessageContent(
                                type=MessageContentType.TOOL_CALL,
                                text=json.dumps(item.get("content", {})),
                                url=None,
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create TOOL_CALL MessageContent: {e}")
                if item.get("type") == "text":
                    try:
                        content.append(
                            MessageContent(
                                type=MessageContentType.TEXT,
                                text=item.get("text", ""),
                                url=None,
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create TEXT MessageContent: {e}")
                elif item.get("type") == "image_url":
                    try:
                        content.append(
                            MessageContent(
                                type=MessageContentType.IMAGE,
                                text=None,
                                url=item.get("image_url", {}).get("url", ""),
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create IMAGE MessageContent: {e}")
                else:
                    # Unknown content type, treat as text
                    try:
                        content.append(
                            MessageContent(
                                type=MessageContentType.TEXT, text=str(item), url=None
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create fallback TEXT MessageContent: {e}")
            else:
                # String content item
                try:
                    content.append(
                        MessageContent(
                            type=MessageContentType.TEXT, text=str(item), url=None
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to create string MessageContent: {e}")
    else:
        # Simple string content
        try:
            content = [
                MessageContent(
                    type=MessageContentType.TEXT,
                    text=str(lc_content) if lc_content else "",
                    url=None,
                )
            ]
        except Exception as e:
            logger.error(f"Failed to create simple text MessageContent: {e}")
            # Fallback to empty list if all else fails
            content = []
    
    # Ensure we always return at least one content item
    if not content:
        logger.warning("No content items created, adding empty text content")
        content = [
            MessageContent(
                type=MessageContentType.TEXT,
                text="",
                url=None,
            )
        ]
    
    return content


def extract_text_from_message(message: Union[Message, BaseMessage]) -> str:
    """
    Extract text content from either a Message object or LangChain BaseMessage.

    This is the unified function that handles both message types.
    """
    if isinstance(message, BaseMessage):
        # Handle LangChain BaseMessage
        if not hasattr(message, "content"):
            return ""

        content = message.content

        # Handle multimodal content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            return "".join(
                text_parts
            )  # Fixed: removed \n join that was causing newline issues

        # Handle simple string content
        return str(content) if content else ""

    else:
        # Handle Message object
        text_parts = []
        for content in message.content:
            # Handle both MessageContent objects and dictionaries
            if isinstance(content, dict):
                # Handle dictionary format: {'type': 'text', 'text': 'content'}
                if content.get("type") == "text" and content.get("text"):
                    text_parts.append(content["text"])
            else:
                # Handle MessageContent object format
                if hasattr(content, "type") and hasattr(content, "text"):
                    if content.type == MessageContentType.TEXT and content.text:
                        text_parts.append(content.text)
        # Fixed: use space join instead of newline to prevent character separation
        return "".join(text_parts)


def get_most_recent_user_message_text(messages: List[BaseMessage]) -> str:
    """Extract text from the most recent user message in a conversation."""
    if not messages:
        return ""

    # Look for the most recent user message
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return extract_text_from_message(msg)

    # Fallback: if no user message found, use the last message
    if messages:
        return extract_text_from_message(messages[-1])

    return ""


def create_text_message_content(text: str) -> List[MessageContent]:
    """
    Create a list containing a single MessageContent object with text content.

    This is the unified function for creating text message content.
    Replaces: convert_string_to_message_content
    """
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
