

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class PipelineMetrics(BaseModel):
    """Performance metrics for pipeline execution"""
    total_tokens: Annotated[Optional[int], Field(default=None, description="Total tokens processed", ge=0)] = None
    """Total tokens processed"""
    processing_time_seconds: Annotated[Optional[float], Field(default=None, description="Total processing time in seconds", ge=0)] = None
    """Total processing time in seconds"""
    memory_usage_mb: Annotated[Optional[float], Field(default=None, description="Peak memory usage in MB", ge=0)] = None
    """Peak memory usage in MB"""
    tool_calls_made: Annotated[Optional[int], Field(default=None, description="Number of tool calls made", ge=0)] = None
    """Number of tool calls made"""
    cache_hits: Annotated[Optional[int], Field(default=None, description="Number of cache hits", ge=0)] = None
    """Number of cache hits"""
    cache_misses: Annotated[Optional[int], Field(default=None, description="Number of cache misses", ge=0)] = None
    """Number of cache misses"""

    model_config = ConfigDict(extra="ignore")