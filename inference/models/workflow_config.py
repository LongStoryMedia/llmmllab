from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat


class WorkflowConfig(BaseModel):
    """User-configurable workflow settings"""

    enable_workflow_caching: Annotated[
        Optional[bool],
        Field(default=True, description="Enable workflow result caching"),
    ] = True
    """Enable workflow result caching"""
    workflow_cache_ttl: Annotated[
        Optional[int],
        Field(
            default=3600, description="Workflow cache TTL in seconds", ge=60, le=86400
        ),
    ] = 3600
    """Workflow cache TTL in seconds"""
    max_parallel_tools: Annotated[
        Optional[int],
        Field(
            default=5,
            description="Maximum number of tools to execute in parallel",
            ge=1,
            le=20,
        ),
    ] = 5
    """Maximum number of tools to execute in parallel"""
    enable_multi_agent: Annotated[
        Optional[bool],
        Field(default=False, description="Enable multi-agent workflow capabilities"),
    ] = False
    """Enable multi-agent workflow capabilities"""
    default_timeout: Annotated[
        Optional[float],
        Field(
            default=60.0,
            description="Default workflow execution timeout in seconds",
            ge=5.0,
            le=300.0,
        ),
    ] = 60.0
    """Default workflow execution timeout in seconds"""
    max_context_length: Annotated[
        Optional[int],
        Field(
            default=128000,
            description="Maximum context length for workflow processing",
            ge=1000,
            le=1000000,
        ),
    ] = 128000
    """Maximum context length for workflow processing"""
    context_trim_threshold: Annotated[
        Optional[float],
        Field(
            default=0.8,
            description="Context trimming threshold (0.0-1.0)",
            ge=0.1,
            le=1.0,
        ),
    ] = 0.8
    """Context trimming threshold (0.0-1.0)"""
    enable_streaming: Annotated[
        Optional[bool],
        Field(default=True, description="Enable streaming workflow responses"),
    ] = True
    """Enable streaming workflow responses"""
    stream_buffer_size: Annotated[
        Optional[int],
        Field(
            default=1024, description="Streaming buffer size in bytes", ge=256, le=8192
        ),
    ] = 1024
    """Streaming buffer size in bytes"""

    model_config = ConfigDict(extra="ignore")
