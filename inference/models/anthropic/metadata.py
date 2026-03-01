from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat


class Metadata(BaseModel):
    user_id: Annotated[Optional[str], Field(default=None)] = None

    model_config = ConfigDict(extra="ignore")
