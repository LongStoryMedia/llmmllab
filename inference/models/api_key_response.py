

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ApiKeyResponse(BaseModel):
    """Response when creating an API key (includes plaintext key once)"""
    id: Annotated[str, Field(..., description="Unique identifier (UUID)")]
    """Unique identifier (UUID)"""
    user_id: Annotated[str, Field(..., description="User ID that owns this key")]
    """User ID that owns this key"""
    key: Annotated[str, Field(..., description="The actual API key (returned only on creation, never stored plaintext)")]
    """The actual API key (returned only on creation, never stored plaintext)"""
    name: Annotated[str, Field(..., description="Human-readable name for this key")]
    """Human-readable name for this key"""
    created_at: Annotated[datetime, Field(..., description="When the key was created")]
    """When the key was created"""
    expires_at: Annotated[Optional[datetime], Field(default=None, description="When the key expires (null = never)")] = None
    """When the key expires (null = never)"""
    scopes: Annotated[List[str], Field(..., description="API scopes this key has access to (e.g., 'chat', 'generate', 'embed')")]
    """API scopes this key has access to (e.g., 'chat', 'generate', 'embed')"""

    model_config = ConfigDict(extra="ignore")