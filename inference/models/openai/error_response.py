

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .error import Error
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ErrorResponse(BaseModel):
    error: Annotated[Error, Field(...)]

    model_config = ConfigDict(extra="ignore")