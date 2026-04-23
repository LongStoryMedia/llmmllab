

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class AuthConfig(BaseModel):
    """Authentication configuration"""
    jwks_uri: Annotated[str, Field(..., description="JWKS URI")]
    """JWKS URI"""
    audience: Annotated[str, Field(..., description="Audience")]
    """Audience"""
    client_id: Annotated[str, Field(..., description="Client ID")]
    """Client ID"""
    client_secret: Annotated[str, Field(..., description="Client secret")]
    """Client secret"""

    model_config = ConfigDict(extra="ignore")