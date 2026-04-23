

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ResearchQuestion(BaseModel):
    """Sub-question in a research plan"""
    id: Annotated[int, Field(..., description="Unique identifier for the question")]
    """Unique identifier for the question"""
    question: Annotated[str, Field(..., description="The question to investigate")]
    """The question to investigate"""
    keywords: Annotated[List[str], Field(..., description="Keywords relevant to the question")]
    """Keywords relevant to the question"""

    model_config = ConfigDict(extra="ignore")