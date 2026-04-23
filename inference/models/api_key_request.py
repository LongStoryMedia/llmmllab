

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ApiKeyRequest(BaseModel):
    """Request to create an API key for IDE and external service integration"""
    name: Annotated[str, Field(..., description="Human-readable name for this key")]
    """Human-readable name for this key"""
    expires_in_days: Annotated[Optional[int], Field(default=None, description="Number of days until the key expires (null = never)")] = None
    """Number of days until the key expires (null = never)"""
    scopes: Annotated[List[str], Field(..., description="API scopes this key has access to (e.g., 'chat', 'generate', 'embed')")]
    """API scopes this key has access to (e.g., 'chat', 'generate', 'embed')"""

    model_config = ConfigDict(extra="ignore")