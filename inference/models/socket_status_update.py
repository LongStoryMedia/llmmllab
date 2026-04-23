

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .socket_stage_type import SocketStageType
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class SocketStatusUpdate(BaseModel):
    """SocketStatusUpdate represents a status update message sent via WebSocket"""
    user_id: Annotated[str, Field(..., description="The ID of the user associated with the status update.")]
    """The ID of the user associated with the status update."""
    type: Annotated[Literal["info", "warning", "error"], Field(..., description="The type of status update (info, warning, error).")]
    """The type of status update (info, warning, error)."""
    stage: Annotated[SocketStageType, Field(..., description="The current processing stage.")]
    """The current processing stage."""
    message: Annotated[Optional[str], Field(default=None, description="Optional detailed message.")] = None
    """Optional detailed message."""
    progress: Annotated[int, Field(..., description="0-100 progress percentage.", ge=0, le=100)]
    """0-100 progress percentage."""
    done: Annotated[bool, Field(..., description="Whether processing is complete.")]
    """Whether processing is complete."""

    model_config = ConfigDict(extra="ignore")