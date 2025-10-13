"""
Message utility functions for validating and formatting Message objects.
"""

from typing import List, Union

from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from models.message import Message
from models.message_role import MessageRole
from models.message_content_type import MessageContentType
from models.message_content import MessageContent
from models.lang_chain_message import LangChainMessage
from .logging import llmmllogger

logger = llmmllogger.bind(module=__name__)


def extract_message_text(message: Message) -> str:
    """Extract text content from a message object"""
    text_parts = []
    for content in message.content:
        if content.type == MessageContentType.TEXT and content.text:
            text_parts.append(content.text)
    # Do not strip whitespace here; streaming tokens often include leading spaces
    # and trimming per-chunk will remove necessary spacing between words.
    return "\n".join(text_parts)


def convert_string_to_message_content(message_text: str) -> List[MessageContent]:
    """Convert a plain string to a list of MessageContent objects"""
    return [MessageContent(type=MessageContentType.TEXT, text=message_text, url=None)]


def ensure_message_content_list(message: Message) -> Message:
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
            except Exception:
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
            except Exception:
                message.content[i] = MessageContent(
                    type=MessageContentType.TEXT, text="", url=None
                )

    return message


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


def from_lc_message(lc_message: Union[BaseMessage, LangChainMessage]) -> Message:
    """Convert a LangChain message or LangChainMessage to a Message object."""

    # Handle generated LangChainMessage objects (from schemas)
    if isinstance(lc_message, LangChainMessage):
        # Use the type field to determine the role
        message_type = lc_message.type.lower() if lc_message.type else ""

        if message_type in ("ai", "assistant"):
            role = MessageRole.ASSISTANT
        elif message_type in ("human", "user"):
            role = MessageRole.USER
        elif message_type == "system":
            role = MessageRole.SYSTEM
        elif message_type == "tool":
            # Tool messages are treated as system messages to preserve context
            role = MessageRole.SYSTEM
        else:
            logger.warning(
                f"Unknown LangChainMessage type: {lc_message.type}, defaulting to USER"
            )
            role = MessageRole.USER

        # Extract content - LangChainMessage.content can be string or list
        if isinstance(lc_message.content, str):
            text_content = lc_message.content
        elif isinstance(lc_message.content, list):
            # Join list items or convert them to string
            text_content = str(lc_message.content)
        else:
            # Handle other types by converting to string
            text_content = str(lc_message.content) if lc_message.content else ""

    # Handle LangChain core BaseMessage objects
    elif isinstance(lc_message, AIMessage):
        role = MessageRole.ASSISTANT
        text_content = str(lc_message.content) if lc_message.content else ""
    elif isinstance(lc_message, HumanMessage):
        role = MessageRole.USER
        text_content = str(lc_message.content) if lc_message.content else ""
    elif isinstance(lc_message, SystemMessage):
        role = MessageRole.SYSTEM
        text_content = str(lc_message.content) if lc_message.content else ""
    elif isinstance(lc_message, ToolMessage):
        # Tool messages are treated as system messages to preserve tool output context
        role = MessageRole.SYSTEM
        text_content = str(lc_message.content) if lc_message.content else ""
    else:
        logger.warning(
            f"Unknown LangChain message type: {type(lc_message)}, defaulting to USER"
        )
        role = MessageRole.USER

        # Extract content for unknown types
        if hasattr(lc_message, "content"):
            if isinstance(lc_message.content, str):
                text_content = lc_message.content
            elif isinstance(lc_message.content, list):
                text_content = str(lc_message.content)
            else:
                text_content = str(lc_message.content) if lc_message.content else ""
        else:
            text_content = str(lc_message)

    return Message(
        role=role,
        content=[
            MessageContent(
                type=MessageContentType.TEXT,
                text=text_content,
                url=None,
            )
        ],
    )
