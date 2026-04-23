

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class WebSocketConnection(BaseModel):
    """WebSocketConnection represents an active WebSocket connection"""
    id: Annotated[str, Field(..., description="Unique identifier for the connection")]
    """Unique identifier for the connection"""
    user_id: Annotated[str, Field(..., description="Identifier of the user associated with the connection")]
    """Identifier of the user associated with the connection"""
    type: Annotated[Literal["chat", "image", "status"], Field(..., description="Type of WebSocket connection (e.g., chat, image, status)")]
    """Type of WebSocket connection (e.g., chat, image, status)"""
    conn: Annotated[Optional[Any], Field(default=None, description="The WebSocket connection object (represented as string)")] = None
    """The WebSocket connection object (represented as string)"""
    created_at: Annotated[datetime, Field(..., description="Timestamp indicating when the connection was established")]
    """Timestamp indicating when the connection was established"""

    model_config = ConfigDict(extra="ignore")