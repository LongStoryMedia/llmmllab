

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class PaginationSchema(BaseModel):
    """Schema defining pagination parameters and metadata for API responses"""
    total_count: Annotated[int, Field(..., description="Total number of items available", ge=0)]
    """Total number of items available"""
    limit: Annotated[int, Field(..., description="Maximum number of items per page", ge=1)]
    """Maximum number of items per page"""
    offset: Annotated[int, Field(..., description="Starting position of the current page", ge=0)]
    """Starting position of the current page"""
    has_more: Annotated[bool, Field(..., description="Indicates whether more items are available beyond the current page")]
    """Indicates whether more items are available beyond the current page"""

    model_config = ConfigDict(extra="ignore")