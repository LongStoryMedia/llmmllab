

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .conversation import Conversation
from .document import Document
from .image_metadata import ImageMetadata
from .intent import Intent
from .message import Message
from .message_content import MessageContent
from .message_content_type import MessageContentType
from .message_role import MessageRole
from .resource_usage import ResourceUsage
from .thought import Thought
from .tool_call import ToolCall
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ConversationCtx(BaseModel):
    """Manages context for a conversation, including messages, summaries, and retrieved memories. Provides methods for adding messages, summarizing conversations, and retrieving relevant context.
"""
    messages: Annotated[List[Message], Field(..., description="List of messages in the conversation")]
    """List of messages in the conversation"""
    notes: Annotated[List[str], Field(..., description="List of notes related to the conversation")]
    """List of notes related to the conversation"""
    images: Annotated[List[ImageMetadata], Field(..., description="List of images associated with the conversation")]
    """List of images associated with the conversation"""
    conversation: Annotated[Conversation, Field(..., description="The conversation object")]
    """The conversation object"""
    current_user_message: Annotated[Optional[Message], Field(default=None, description="The current message from the user")] = None
    """The current message from the user"""
    intent: Annotated[Optional[Intent], Field(default=None, description="The intent detected from the user's message")] = None
    """The intent detected from the user's message"""

    model_config = ConfigDict(extra="ignore")