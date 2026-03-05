"""
Helper functions for message processing.
"""

import datetime
from typing import Optional, Any, Dict

from models import (
    MessageContent,
    MessageContentType,
    MessageRole,
    Message,
    ChatResponse,
)

from .logging import llmmllogger

logger = llmmllogger.bind(module=__name__)


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
            role=MessageRole.OBSERVER,
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
