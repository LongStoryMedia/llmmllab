

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .task_type import TaskType
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class TaskGroupTask(BaseModel):
    """Task entry that appears within a TaskGroup."""
    heading: Annotated[Union[str, Any], Field(...)]
    summary: Annotated[Union[str, Any], Field(...)]
    type: Annotated[TaskType, Field(..., description="Subtype for the grouped task.")]
    """Subtype for the grouped task."""

    model_config = ConfigDict(extra="ignore")