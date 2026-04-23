

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class SearchResultContent(BaseModel):
    """Content item from a web search result"""
    url: Annotated[str, Field(..., description="URL of the content")]
    """URL of the content"""
    title: Annotated[str, Field(..., description="Title of the content")]
    """Title of the content"""
    content: Annotated[str, Field(..., description="Snippet or summary of the content")]
    """Snippet or summary of the content"""
    relevance: Annotated[Optional[float], Field(default=None, description="Relevance score of the content")] = None
    """Relevance score of the content"""

    model_config = ConfigDict(extra="ignore")