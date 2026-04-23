

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .cache_control import CacheControl
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat


class ToolReferenceContentBlock(BaseModel):
    """Tool reference block that can be included in tool_result content."""

    type: Annotated[Literal["tool_reference"], Field(...)]
    tool_name: Annotated[str, Field(...)]
    cache_control: Annotated[Optional[CacheControl], Field(default=None)] = None

    model_config = ConfigDict(extra="ignore")
