

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .memory_fragment import MemoryFragment
from .memory_source import MemorySource
from .message_role import MessageRole
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class Memory(BaseModel):
    """Memory represents a grouped memory for a user, which can be a summary or a question-answer pair"""
    fragments: Annotated[List[MemoryFragment], Field(..., description="Fragments of memory content")]
    """Fragments of memory content"""
    source: Annotated[MemorySource, Field(..., description="Source of the memory")]
    """Source of the memory"""
    created_at: Annotated[datetime, Field(..., description="Timestamp of when the memory was created")]
    """Timestamp of when the memory was created"""
    similarity: Annotated[float, Field(..., description="Similarity score from vector search")]
    """Similarity score from vector search"""
    source_id: Annotated[int, Field(..., description="Id of the source document or conversation")]
    """Id of the source document or conversation"""
    conversation_id: Annotated[Optional[int], Field(default=None, description="Id of the conversation this memory belongs to")] = None
    """Id of the conversation this memory belongs to"""

    model_config = ConfigDict(extra="ignore")