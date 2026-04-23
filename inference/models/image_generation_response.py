

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ImageGenerateResponse(BaseModel):
    """Response from generating an image"""
    image: Annotated[str, Field(..., description="Base64-encoded image")]
    """Base64-encoded image"""
    download: Annotated[str, Field(..., description="URL to download the image")]
    """URL to download the image"""

    model_config = ConfigDict(extra="ignore")