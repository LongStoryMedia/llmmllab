

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .socket_stage_type import SocketStageType
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class SocketSession(BaseModel):
    """SocketSession represents the state of a chat session over WebSocket."""
    state: Annotated[SocketStageType, Field(..., description="The current stage/state of the chat session (models.SocketStageType)")]
    """The current stage/state of the chat session (models.SocketStageType)"""
    conversation_id: Annotated[int, Field(..., description="The ID of the conversation")]
    """The ID of the conversation"""
    user_id: Annotated[str, Field(..., description="The user ID associated with the session")]
    """The user ID associated with the session"""
    cancel_func: Annotated[Optional[Any], Field(default=None, description="Reference to the context cancellation function (not serializable, for documentation only)")] = None
    """Reference to the context cancellation function (not serializable, for documentation only)"""
    paused_content: Annotated[Optional[str], Field(default=None, description="Content paused in the session")] = None
    """Content paused in the session"""
    last_message: Annotated[Optional[str], Field(default=None, description="The last message sent or received")] = None
    """The last message sent or received"""
    response_buffer: Annotated[Optional[str], Field(default=None, description="Buffer for the response content")] = None
    """Buffer for the response content"""

    model_config = ConfigDict(extra="ignore")