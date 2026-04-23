

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .cache_control import CacheControl
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class TextContentBlock(BaseModel):
    type: Annotated[Literal["text"], Field(...)]
    text: Annotated[str, Field(...)]
    cache_control: Annotated[Optional[CacheControl], Field(default=None)] = None

    model_config = ConfigDict(extra="ignore")