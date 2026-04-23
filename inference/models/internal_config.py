

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class InternalConfig(BaseModel):
    """Configuration for internal services, including API keys and allowed IPs."""
    APIKey: Annotated[str, Field(..., description="API key for internal service auth")]
    """API key for internal service auth"""
    AllowedIPs: Annotated[str, Field(..., description="Comma-separated list of allowed IPs")]
    """Comma-separated list of allowed IPs"""

    model_config = ConfigDict(extra="ignore")