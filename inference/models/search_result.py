

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .search_result_content import SearchResultContent
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class SearchResult(BaseModel):
    """Search result from a web query"""
    is_from_url_in_user_query: Annotated[bool, Field(..., description="Indicates if this result is from a URL specified in the user query")]
    """Indicates if this result is from a URL specified in the user query"""
    query: Annotated[str, Field(..., description="The search query that produced these results")]
    """The search query that produced these results"""
    contents: Annotated[Optional[List[SearchResultContent]], Field(default=None, description="Content items found in the search")] = None
    """Content items found in the search"""
    error: Annotated[Optional[str], Field(default=None, description="Error message if the search failed")] = None
    """Error message if the search failed"""

    model_config = ConfigDict(extra="ignore")