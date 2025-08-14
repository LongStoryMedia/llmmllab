# Pre-validation hook for Message class
# This enhances the Message class with stronger validation for content and conversation_id fields

from typing import List, Dict, Any, Union, Optional, Type
import logging
from pydantic import validator, root_validator

# Import the required model classes
from models.message import Message
from models.message_content import MessageContent
from models.message_content_type import MessageContentType

logger = logging.getLogger(__name__)


# Add custom validator methods to Message class
@validator("content", pre=True)
def ensure_content_is_list(cls, v):
    """Ensure content is always a list of MessageContent objects"""
    if not v:
        return [MessageContent(type=MessageContentType.TEXT, text="")]

    if not isinstance(v, list):
        return [MessageContent(type=MessageContentType.TEXT, text=str(v))]

    # If it's a list but empty
    if len(v) == 0:
        return [MessageContent(type=MessageContentType.TEXT, text="")]

    # If it contains items that aren't MessageContent objects
    result = []
    for item in v:
        if isinstance(item, MessageContent):
            result.append(item)
        else:
            text = str(item) if item is not None else ""
            result.append(MessageContent(type=MessageContentType.TEXT, text=text))

    return result


@validator("conversation_id")
def ensure_conversation_id(cls, v):
    """Ensure conversation_id is always set to a valid value"""
    if v is None:
        return -1
    return v


# Attach validators to the Message class
if not hasattr(Message, "_validators_added"):
    # Add the validators to the Message class
    setattr(Message, "ensure_content_is_list", ensure_content_is_list)
    setattr(Message, "ensure_conversation_id", ensure_conversation_id)

    # Mark the class so we don't add the validators twice
    setattr(Message, "_validators_added", True)

    # Log success
    logger.info(
        "Added custom validators to Message class for improved content and conversation_id validation"
    )
