

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ApiKey(BaseModel):
    """API key for IDE and external service integration"""
    id: Annotated[str, Field(..., description="Unique identifier (UUID)")]
    """Unique identifier (UUID)"""
    user_id: Annotated[str, Field(..., description="User ID that owns this key")]
    """User ID that owns this key"""
    key_hash: Annotated[str, Field(..., description="SHA-256 hash of the actual API key (never store plaintext)")]
    """SHA-256 hash of the actual API key (never store plaintext)"""
    name: Annotated[str, Field(..., description="Human-readable name for this key")]
    """Human-readable name for this key"""
    created_at: Annotated[datetime, Field(..., description="When the key was created")]
    """When the key was created"""
    last_used_at: Annotated[Optional[datetime], Field(default=None, description="When the key was last used")] = None
    """When the key was last used"""
    expires_at: Annotated[Optional[datetime], Field(default=None, description="When the key expires (null = never)")] = None
    """When the key expires (null = never)"""
    is_revoked: Annotated[Optional[bool], Field(default=None, description="Whether this key has been revoked")] = None
    """Whether this key has been revoked"""
    scopes: Annotated[List[str], Field(..., description="API scopes this key has access to (e.g., 'chat', 'generate', 'embed')")]
    """API scopes this key has access to (e.g., 'chat', 'generate', 'embed')"""

    model_config = ConfigDict(extra="ignore")