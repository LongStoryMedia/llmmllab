

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .static_chunking_strategy import StaticChunkingStrategy
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class StaticChunkingStrategyResponseParam(BaseModel):
    static: Annotated[StaticChunkingStrategy, Field(...)]
    type: Annotated[Literal["static"], Field(..., description="Always `static`.")]
    """Always `static`."""

    model_config = ConfigDict(extra="ignore")