

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .message_role import MessageRole
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class MemoryFragment(BaseModel):
    """Content item that is similar to a user's query"""
    id: Annotated[int, Field(..., description="Unique identifier for the memory fragment")]
    """Unique identifier for the memory fragment"""
    role: Annotated[MessageRole, Field(..., description="Role of the message (e.g., 'user', 'assistant')")]
    """Role of the message (e.g., 'user', 'assistant')"""
    content: Annotated[Optional[str], Field(default=None, description="Content of the message")] = None
    """Content of the message"""
    embeddings: Annotated[Optional[List[List[float]]], Field(default=None, description="Embedding vector representing the memory")] = None
    """Embedding vector representing the memory"""

    model_config = ConfigDict(extra="ignore")