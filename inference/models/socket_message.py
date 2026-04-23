

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .message_type import MessageType
from .socket_stage_type import SocketStageType
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class SocketMessage(BaseModel):
    """SocketMessage represents a response sent over WebSocket"""
    id: Annotated[str, Field(..., description="Unique identifier for the message")]
    """Unique identifier for the message"""
    content: Annotated[Optional[str], Field(default=None, description="Content of the response")] = None
    """Content of the response"""
    error: Annotated[Optional[str], Field(default=None, description="Error message if any")] = None
    """Error message if any"""
    state: Annotated[SocketStageType, Field(..., description="State of the chat session")]
    """State of the chat session"""
    session_id: Annotated[str, Field(..., description="Session identifier")]
    """Session identifier"""
    timestamp: Annotated[datetime, Field(..., description="Timestamp of the response")]
    """Timestamp of the response"""
    type: Annotated[MessageType, Field(..., description="Type of message")]
    """Type of message"""
    progress: Annotated[Optional[int], Field(default=None, description="Progress percentage of the response, if applicable", ge=0, le=100)] = None
    """Progress percentage of the response, if applicable"""
    data: Annotated[Optional[List[Any]], Field(default=None, description="Additional data related to the response, if any")] = None
    """Additional data related to the response, if any"""
    conversation_id: Annotated[Optional[int], Field(default=None, description="Identifier of the conversation associated with the response")] = None
    """Identifier of the conversation associated with the response"""

    model_config = ConfigDict(extra="ignore")