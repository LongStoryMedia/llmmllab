

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ToolChoice(BaseModel):
    """Tool selection that the assistant should honor when executing the item."""
    id: Annotated[str, Field(..., description="Identifier of the requested tool.")]
    """Identifier of the requested tool."""

    model_config = ConfigDict(extra="ignore")