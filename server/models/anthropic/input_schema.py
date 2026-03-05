from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat


class InputSchema(BaseModel):
    """JSON Schema for the tool's input."""

    type: Annotated[Literal["object"], Field(...)]
    properties: Annotated[Optional[Dict[str, Any]], Field(default=None)] = None
    required: Annotated[Optional[List[str]], Field(default=None)] = None

    model_config = ConfigDict(extra="ignore")
