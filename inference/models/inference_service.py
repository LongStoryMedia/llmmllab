

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class InferenceService(BaseModel):
    """Inference service configuration"""
    base_url: Annotated[str, Field(..., description="Base URL for the inference service")]
    """Base URL for the inference service"""

    model_config = ConfigDict(extra="ignore")