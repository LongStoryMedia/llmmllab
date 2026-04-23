

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .research_question import ResearchQuestion
from .research_question_result import ResearchQuestionResult
from .research_subtask import ResearchSubtask
from .research_task_status import ResearchTaskStatus
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ResearchTask(BaseModel):
    """ResearchTask represents a deep research task"""
    id: Annotated[int, Field(..., description="Unique identifier for the research task")]
    """Unique identifier for the research task"""
    user_id: Annotated[str, Field(..., description="Id of the user who owns the task")]
    """Id of the user who owns the task"""
    query: Annotated[str, Field(..., description="Original research query")]
    """Original research query"""
    conversation_id: Annotated[Optional[int], Field(default=None, description="Id of the conversation associated with this task if any")] = None
    """Id of the conversation associated with this task if any"""
    status: Annotated[ResearchTaskStatus, Field(..., description="Current status of the research task")]
    """Current status of the research task"""
    error_message: Annotated[Optional[str], Field(default=None, description="Error message if task failed")] = None
    """Error message if task failed"""
    plan: Annotated[Optional[str], Field(default=None, description="Research plan with sub-questions")] = None
    """Research plan with sub-questions"""
    embedding: Annotated[Optional[List[float]], Field(default=None, description="The reasearch plan's embedding vector")] = None
    """The reasearch plan's embedding vector"""
    subtasks: Annotated[Optional[List[ResearchSubtask]], Field(default=None, description="List of subtasks associated with this research task")] = None
    """List of subtasks associated with this research task"""
    created_at: Annotated[datetime, Field(..., description="Timestamp of when the task was created")]
    """Timestamp of when the task was created"""
    updated_at: Annotated[datetime, Field(..., description="Timestamp of when the task was last updated")]
    """Timestamp of when the task was last updated"""
    completed_at: Annotated[Optional[datetime], Field(default=None, description="Timestamp of when the task was completed")] = None
    """Timestamp of when the task was completed"""

    model_config = ConfigDict(extra="ignore")