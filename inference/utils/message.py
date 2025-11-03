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
        # Handle both MessageContent objects and dictionaries
        if isinstance(content, dict):
            # Handle dictionary format: {'type': 'text', 'text': 'content'}
            if content.get('type') == 'text' and content.get('text'):
                text_parts.append(content['text'])
        else:
            # Handle MessageContent object format
            if hasattr(content, 'type') and hasattr(content, 'text'):
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
