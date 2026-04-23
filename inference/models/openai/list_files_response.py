

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .open_ai_file import OpenAIFile
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ListFilesResponse(BaseModel):
    data: Annotated[List[OpenAIFile], Field(...)]
    first_id: Annotated[str, Field(...)]
    has_more: Annotated[bool, Field(...)]
    last_id: Annotated[str, Field(...)]
    object: Annotated[str, Field(...)]

    model_config = ConfigDict(extra="ignore")