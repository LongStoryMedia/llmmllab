

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class Conversation(BaseModel):
    """Conversation represents a conversation between a user and the LLM"""
    id: Annotated[int, Field(..., description="Unique identifier for the conversation")]
    """Unique identifier for the conversation"""
    user_id: Annotated[str, Field(..., description="Id of the user who owns the conversation")]
    """Id of the user who owns the conversation"""
    title: Annotated[str, Field(..., description="Title of the conversation")]
    """Title of the conversation"""
    created_at: Annotated[datetime, Field(..., description="Timestamp of when the conversation was created")]
    """Timestamp of when the conversation was created"""
    updated_at: Annotated[datetime, Field(..., description="Timestamp of when the conversation was last updated")]
    """Timestamp of when the conversation was last updated"""

    model_config = ConfigDict(extra="ignore")