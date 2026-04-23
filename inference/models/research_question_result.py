

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ResearchQuestionResult(BaseModel):
    """Result of processing a sub-question in a research task"""
    id: Annotated[int, Field(..., description="Unique identifier for the question result")]
    """Unique identifier for the question result"""
    sources: Annotated[Optional[List[str]], Field(default=None, description="List of sources used to gather information")] = None
    """List of sources used to gather information"""
    synthesized_answer: Annotated[Optional[str], Field(default=None, description="Synthesized answer to the question")] = None
    """Synthesized answer to the question"""
    error_message: Annotated[Optional[str], Field(default=None, description="Error message if processing failed")] = None
    """Error message if processing failed"""

    model_config = ConfigDict(extra="ignore")