

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class Summary(BaseModel):
    """Summary represents a consolidated summary of messages or other summaries"""
    content: Annotated[str, Field(..., description="Text content of the summary")]
    """Text content of the summary"""
    level: Annotated[int, Field(..., description="Level of the summary in the hierarchy")]
    """Level of the summary in the hierarchy"""
    id: Annotated[int, Field(..., description="Unique identifier for the summary")]
    """Unique identifier for the summary"""
    conversation_id: Annotated[int, Field(..., description="Id of the conversation this summary belongs to")]
    """Id of the conversation this summary belongs to"""
    source_ids: Annotated[List[int], Field(..., description="IDs of the source messages or summaries")]
    """IDs of the source messages or summaries"""
    created_at: Annotated[datetime, Field(..., description="Timestamp of when the summary was created")]
    """Timestamp of when the summary was created"""

    model_config = ConfigDict(extra="ignore")