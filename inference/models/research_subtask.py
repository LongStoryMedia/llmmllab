

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .research_question import ResearchQuestion
from .research_question_result import ResearchQuestionResult
from .research_task_status import ResearchTaskStatus
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ResearchSubtask(BaseModel):
    """Subtask within a research task"""
    id: Annotated[int, Field(..., description="Unique identifier for the subtask")]
    """Unique identifier for the subtask"""
    task_id: Annotated[int, Field(..., description="Id of the parent research task")]
    """Id of the parent research task"""
    question: Annotated[Optional[ResearchQuestion], Field(default=None, description="The research question this subtask addresses")] = None
    """The research question this subtask addresses"""
    result: Annotated[Optional[ResearchQuestionResult], Field(default=None, description="Synthesized result of the subtask")] = None
    """Synthesized result of the subtask"""
    status: Annotated[ResearchTaskStatus, Field(..., description="Current status of the subtask")]
    """Current status of the subtask"""
    error_message: Annotated[Optional[str], Field(default=None, description="Error message if subtask failed")] = None
    """Error message if subtask failed"""
    created_at: Annotated[datetime, Field(..., description="Timestamp of when the subtask was created")]
    """Timestamp of when the subtask was created"""
    updated_at: Annotated[datetime, Field(..., description="Timestamp of when the subtask was last updated")]
    """Timestamp of when the subtask was last updated"""

    model_config = ConfigDict(extra="ignore")