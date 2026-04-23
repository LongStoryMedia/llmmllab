

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class User(BaseModel):
    """User represents a user of the system"""
    id: Annotated[str, Field(..., description="Unique identifier for the user")]
    """Unique identifier for the user"""
    username: Annotated[str, Field(..., description="Username of the user")]
    """Username of the user"""
    uid: Annotated[str, Field(..., description="User identification number")]
    """User identification number"""
    created_at: Annotated[datetime, Field(..., description="Timestamp of when the user was created")]
    """Timestamp of when the user was created"""

    model_config = ConfigDict(extra="ignore")