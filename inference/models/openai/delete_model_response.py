

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class DeleteModelResponse(BaseModel):
    deleted: Annotated[bool, Field(...)]
    id: Annotated[str, Field(...)]
    object: Annotated[str, Field(...)]

    model_config = ConfigDict(extra="ignore")