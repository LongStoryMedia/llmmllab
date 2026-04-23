

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class DatabaseConfig(BaseModel):
    """Database configuration"""
    host: Annotated[str, Field(..., description="Database host")]
    """Database host"""
    port: Annotated[int, Field(..., description="Database port")]
    """Database port"""
    user: Annotated[str, Field(..., description="Database user")]
    """Database user"""
    password: Annotated[str, Field(..., description="Database password")]
    """Database password"""
    dbname: Annotated[str, Field(..., description="Database name")]
    """Database name"""
    sslmode: Annotated[str, Field(..., description="SSL mode")]
    """SSL mode"""
    connection_string: Annotated[str, Field(..., description="Connection string")]
    """Connection string"""

    model_config = ConfigDict(extra="ignore")