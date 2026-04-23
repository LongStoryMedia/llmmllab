

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ProjectServiceAccountCreateRequest(BaseModel):
    name: Annotated[str, Field(..., description="The name of the service account being created.")]
    """The name of the service account being created."""

    model_config = ConfigDict(extra="ignore")