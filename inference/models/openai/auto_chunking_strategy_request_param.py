

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class AutoChunkingStrategyRequestParam(BaseModel):
    """The default strategy. This strategy currently uses a `max_chunk_size_tokens` of `800` and `chunk_overlap_tokens` of `400`."""
    type: Annotated[Literal["auto"], Field(..., description="Always `auto`.")]
    """Always `auto`."""

    model_config = ConfigDict(extra="ignore")