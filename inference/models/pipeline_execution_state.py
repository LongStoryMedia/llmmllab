

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .pipeline_metrics import PipelineMetrics
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class PipelineExecutionState(BaseModel):
    """State tracking for pipeline execution"""
    pipeline_id: Annotated[str, Field(..., description="Unique identifier for the pipeline execution")]
    """Unique identifier for the pipeline execution"""
    conversation_id: Annotated[int, Field(..., description="Associated conversation ID")]
    """Associated conversation ID"""
    user_id: Annotated[str, Field(..., description="User ID for the execution")]
    """User ID for the execution"""
    status: Annotated[Literal["initializing", "running", "completed", "failed", "cancelled"], Field(..., description="Current execution status")]
    """Current execution status"""
    start_time: Annotated[datetime, Field(..., description="When execution started")]
    """When execution started"""
    end_time: Annotated[Optional[datetime], Field(default=None, description="When execution completed")] = None
    """When execution completed"""
    error_message: Annotated[Optional[str], Field(default=None, description="Error message if execution failed")] = None
    """Error message if execution failed"""
    metrics: Annotated[Optional[PipelineMetrics], Field(default=None)] = None

    model_config = ConfigDict(extra="ignore")