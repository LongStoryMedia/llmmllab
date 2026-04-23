

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .execution_state import ExecutionState
from .resource_usage import ResourceUsage
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class PipelineExecutionContext(BaseModel):
    """Context for tracking pipeline execution lifecycle"""
    pipeline_id: Annotated[str, Field(..., description="Unique identifier for the pipeline execution")]
    """Unique identifier for the pipeline execution"""
    task_name: Annotated[str, Field(..., description="Name of the task being executed")]
    """Name of the task being executed"""
    user_id: Annotated[str, Field(..., description="ID of the user initiating the pipeline")]
    """ID of the user initiating the pipeline"""
    model_id: Annotated[str, Field(..., description="ID of the model being used")]
    """ID of the model being used"""
    profile_id: Annotated[str, Field(..., description="ID of the model profile being used")]
    """ID of the model profile being used"""
    task_type: Annotated[str, Field(..., description="Type of task being executed")]
    """Type of task being executed"""
    arguments: Annotated[Dict[str, Any], Field(..., description="Arguments passed to the pipeline")]
    """Arguments passed to the pipeline"""
    start_time: Annotated[datetime, Field(..., description="When the pipeline execution started")]
    """When the pipeline execution started"""
    end_time: Annotated[Optional[datetime], Field(default=None, description="When the pipeline execution ended")] = None
    """When the pipeline execution ended"""
    execution_state: Annotated[ExecutionState, Field(..., description="Current execution state")]
    """Current execution state"""
    result: Annotated[Optional[Dict[str, Any]], Field(default=None, description="Pipeline execution result")] = None
    """Pipeline execution result"""
    error_message: Annotated[Optional[str], Field(default=None, description="Error message if execution failed")] = None
    """Error message if execution failed"""
    timeout_seconds: Annotated[Optional[float], Field(default=None, description="Timeout for pipeline execution in seconds", ge=0.1)] = None
    """Timeout for pipeline execution in seconds"""
    resource_usage: Annotated[Optional[ResourceUsage], Field(default=None, description="Resource usage tracking")] = None
    """Resource usage tracking"""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())