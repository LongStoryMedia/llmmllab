

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class CreateVideoRemixBody(BaseModel):
    """Parameters for remixing an existing generated video."""
    prompt: Annotated[str, Field(..., description="Updated text prompt that directs the remix generation.")]
    """Updated text prompt that directs the remix generation."""

    model_config = ConfigDict(extra="ignore")