"""
Helper functions for message processing.
"""

import datetime
import logging
from typing import Optional

from models import (
    MessageContent,
    MessageContentType,
    MessageRole,
    Message,
    ChatResponse,
)

logger = logging.getLogger(__name__)


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


def create_error_chunk(error_message: str) -> ChatResponse:
    """Create an error chunk as a ChatResponse."""
    return create_error_response(error_message)


def create_streaming_chunk_with_thinking(
    text: Optional[str] = None, 
    thinking: Optional[str] = None, 
    done: bool = False, 
    role: MessageRole = MessageRole.ASSISTANT
) -> ChatResponse:
    """Create streaming chunk with separate thinking and content routing for harmony channels."""
    message = None
    
    # Only create message if we have content or thinking, or if not done
    if text or thinking or not done:
        content_list = []
        if text:
            content_list.append(MessageContent(type=MessageContentType.TEXT, text=text))
        
        message = Message(
            role=role,
            content=content_list,
            thinking=thinking,  # Route analysis channel content here
        )

    return ChatResponse(
        done=done,
        message=message,
        thinking=thinking,  # Also set at response level for compatibility
        created_at=datetime.datetime.now(datetime.timezone.utc),
        finish_reason="stop" if done else None,
    )
