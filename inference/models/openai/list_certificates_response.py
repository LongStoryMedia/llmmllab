

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .certificate import Certificate
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ListCertificatesResponse(BaseModel):
    data: Annotated[List[Certificate], Field(...)]
    first_id: Annotated[Optional[str], Field(default=None)] = None
    has_more: Annotated[bool, Field(...)]
    last_id: Annotated[Optional[str], Field(default=None)] = None
    object: Annotated[Literal["list"], Field(...)]

    model_config = ConfigDict(extra="ignore")