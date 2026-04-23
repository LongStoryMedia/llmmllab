

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class RedisConfig(BaseModel):
    """Redis configuration"""
    host: Annotated[str, Field(..., description="Redis host")]
    """Redis host"""
    port: Annotated[int, Field(..., description="Redis port")]
    """Redis port"""
    password: Annotated[str, Field(..., description="Redis password")]
    """Redis password"""
    db: Annotated[int, Field(..., description="Redis database index")]
    """Redis database index"""
    enabled: Annotated[bool, Field(..., description="Whether Redis is enabled")]
    """Whether Redis is enabled"""
    conversation_ttl: Annotated[int, Field(..., description="Conversation TTL")]
    """Conversation TTL"""
    message_ttl: Annotated[int, Field(..., description="Message TTL")]
    """Message TTL"""
    summary_ttl: Annotated[int, Field(..., description="Summary TTL")]
    """Summary TTL"""
    pool_size: Annotated[int, Field(..., description="Connection pool size")]
    """Connection pool size"""
    min_idle_connections: Annotated[int, Field(..., description="Minimum idle connections")]
    """Minimum idle connections"""
    connect_timeout: Annotated[int, Field(..., description="Connection timeout")]
    """Connection timeout"""

    model_config = ConfigDict(extra="ignore")