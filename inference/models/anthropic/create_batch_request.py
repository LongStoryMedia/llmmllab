

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat

from .batch_request import BatchRequest


class CreateBatchRequest(BaseModel):
    requests: Annotated[
        List[BatchRequest], Field(..., description="Up to 100,000 requests per batch.")
    ]
    """Up to 100,000 requests per batch."""

    model_config = ConfigDict(extra="ignore")


CreateBatchRequest.model_rebuild()
