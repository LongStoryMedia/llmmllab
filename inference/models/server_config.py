

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ServerConfig(BaseModel):
    """Server configuration"""
    host: Annotated[str, Field(..., description="Server host")]
    """Server host"""
    port: Annotated[int, Field(..., description="Server port")]
    """Server port"""
    base_url: Annotated[str, Field(..., description="Base URL")]
    """Base URL"""

    model_config = ConfigDict(extra="ignore")