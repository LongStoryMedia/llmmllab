

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class SearchTopicSynthesis(BaseModel):
    """Schema for synthesizing search topics"""
    urls: Annotated[List[str], Field(..., description="List of URLs used to synthesize topics")]
    """List of URLs used to synthesize topics"""
    topics: Annotated[List[str], Field(..., description="List of topics synthesized from the URLs")]
    """List of topics synthesized from the URLs"""
    synthesis: Annotated[str, Field(..., description="The summary or synthesis of the topics from the content at the provided URLs")]
    """The summary or synthesis of the topics from the content at the provided URLs"""
    created_at: Annotated[datetime, Field(..., description="The timestamp when the synthesis was created")]
    """The timestamp when the synthesis was created"""
    id: Annotated[Optional[int], Field(default=None, description="The unique identifier for the synthesis")] = None
    """The unique identifier for the synthesis"""
    conversation_id: Annotated[int, Field(..., description="The ID of the conversation associated with this synthesis")]
    """The ID of the conversation associated with this synthesis"""

    model_config = ConfigDict(extra="ignore")