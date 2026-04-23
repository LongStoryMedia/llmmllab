

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class GenerateResponse(BaseModel):
    """GenerateResponse represents a request to the Ollama embeddings API"""
    model: Annotated[str, Field(...)]
    created_at: Annotated[Optional[str], Field(default=None)] = None
    response: Annotated[str, Field(...)]
    done: Annotated[bool, Field(...)]
    done_reason: Annotated[Optional[str], Field(default=None)] = None
    context: Annotated[Optional[List[int]], Field(default=None, description="context IDs for the response")] = None
    """context IDs for the response"""
    prompt_eval_count: Annotated[Optional[int], Field(default=None)] = None
    prompt_eval_duration: Annotated[Optional[float], Field(default=None)] = None
    eval_count: Annotated[Optional[int], Field(default=None)] = None
    eval_duration: Annotated[Optional[float], Field(default=None)] = None
    total_duration: Annotated[Optional[float], Field(default=None)] = None
    load_duration: Annotated[Optional[float], Field(default=None)] = None

    model_config = ConfigDict(extra="ignore")