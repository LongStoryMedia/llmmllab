

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .project_service_account import ProjectServiceAccount
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ProjectServiceAccountListResponse(BaseModel):
    data: Annotated[List[ProjectServiceAccount], Field(...)]
    first_id: Annotated[str, Field(...)]
    has_more: Annotated[bool, Field(...)]
    last_id: Annotated[str, Field(...)]
    object: Annotated[Literal["list"], Field(...)]

    model_config = ConfigDict(extra="ignore")