

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .project import Project
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ProjectListResponse(BaseModel):
    data: Annotated[List[Project], Field(...)]
    first_id: Annotated[str, Field(...)]
    has_more: Annotated[bool, Field(...)]
    last_id: Annotated[str, Field(...)]
    object: Annotated[Literal["list"], Field(...)]

    model_config = ConfigDict(extra="ignore")