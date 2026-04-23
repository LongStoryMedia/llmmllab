

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ProjectUserUpdateRequest(BaseModel):
    role: Annotated[Literal["owner", "member"], Field(..., description="`owner` or `member`")]
    """`owner` or `member`"""

    model_config = ConfigDict(extra="ignore")