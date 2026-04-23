from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .event_stream_config import EventStreamConfig
from .image_generation_config import ImageGenerationConfig
from .memory_config import MemoryConfig
from .summarization_config import SummarizationConfig
from .tool_config import ToolConfig
from .workflow_config import WorkflowConfig
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat


class UserConfig(BaseModel):
    """User-specific configuration"""

    user_id: Annotated[str, Field(..., description="User ID")]
    """User ID"""
    summarization: Annotated[Optional[SummarizationConfig], Field(default=None)] = None
    memory: Annotated[Optional[MemoryConfig], Field(default=None)] = None
    image_generation: Annotated[
        Optional[ImageGenerationConfig], Field(default=None)
    ] = None
    workflow: Annotated[WorkflowConfig, Field(default=WorkflowConfig())]
    tool: Annotated[Optional[ToolConfig], Field(default=None)] = None
    event_stream: Annotated[Optional[EventStreamConfig], Field(default=None)] = None

    model_config = ConfigDict(extra="ignore", protected_namespaces=())
