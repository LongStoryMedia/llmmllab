

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat

from .error_detail import ErrorDetail


class ErrorResponse(BaseModel):
    type: Annotated[Literal["error"], Field(...)]
    error: Annotated[ErrorDetail, Field(...)]

    model_config = ConfigDict(extra="ignore")


ErrorResponse.model_rebuild()
