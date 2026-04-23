

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .project_rate_limit import ProjectRateLimit
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ProjectRateLimitListResponse(BaseModel):
    data: Annotated[List[ProjectRateLimit], Field(...)]
    first_id: Annotated[str, Field(...)]
    has_more: Annotated[bool, Field(...)]
    last_id: Annotated[str, Field(...)]
    object: Annotated[Literal["list"], Field(...)]

    model_config = ConfigDict(extra="ignore")