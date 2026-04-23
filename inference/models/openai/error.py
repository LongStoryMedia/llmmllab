

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class Error(BaseModel):
    code: Annotated[Union[str, Any], Field(...)]
    message: Annotated[str, Field(...)]
    param: Annotated[Union[str, Any], Field(...)]
    type: Annotated[str, Field(...)]

    model_config = ConfigDict(extra="ignore")