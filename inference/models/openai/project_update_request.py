

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ProjectUpdateRequest(BaseModel):
    name: Annotated[str, Field(..., description="The updated name of the project, this name appears in reports.")]
    """The updated name of the project, this name appears in reports."""

    model_config = ConfigDict(extra="ignore")