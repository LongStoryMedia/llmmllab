

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class Intent(BaseModel):
    """Represents the intent detected from a user's message"""
    web_search: Annotated[Optional[bool], Field(default=None, description="Indicates if the intent involves a web search")] = None
    """Indicates if the intent involves a web search"""
    memory: Annotated[Optional[bool], Field(default=None, description="Indicates if the intent involves memory retrieval")] = None
    """Indicates if the intent involves memory retrieval"""
    deep_research: Annotated[Optional[bool], Field(default=None, description="Indicates if the intent involves deep research")] = None
    """Indicates if the intent involves deep research"""
    image_generation: Annotated[Optional[bool], Field(default=None, description="Indicates if the intent involves image generation")] = None
    """Indicates if the intent involves image generation"""
    statically_discovered: Annotated[Optional[bool], Field(default=None, description="Indicates if the intent was statically discovered")] = None
    """Indicates if the intent was statically discovered"""

    model_config = ConfigDict(extra="ignore")