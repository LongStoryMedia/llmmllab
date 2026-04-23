

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class LangGraphNodeState(BaseModel):
    """State for individual nodes in LangGraph workflows"""
    node_id: Annotated[str, Field(..., description="Unique identifier for the node")]
    """Unique identifier for the node"""
    node_type: Annotated[Literal["agent", "tool", "decision", "memory", "search"], Field(..., description="Type of the node")]
    """Type of the node"""
    input_data: Annotated[Optional[Dict[str, Any]], Field(default=None, description="Input data for the node")] = None
    """Input data for the node"""
    output_data: Annotated[Optional[Dict[str, Any]], Field(default=None, description="Output data from the node")] = None
    """Output data from the node"""
    execution_time_ms: Annotated[Optional[float], Field(default=None, description="Time taken to execute this node", ge=0)] = None
    """Time taken to execute this node"""
    error_count: Annotated[Optional[int], Field(default=None, description="Number of errors encountered", ge=0)] = None
    """Number of errors encountered"""
    retry_count: Annotated[Optional[int], Field(default=None, description="Number of retries attempted", ge=0)] = None
    """Number of retries attempted"""

    model_config = ConfigDict(extra="ignore")