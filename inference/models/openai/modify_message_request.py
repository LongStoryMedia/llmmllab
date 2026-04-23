

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .metadata import Metadata
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ModifyMessageRequest(BaseModel):
    metadata: Annotated[Optional[Metadata], Field(default=None)] = None

    model_config = ConfigDict(extra="ignore")