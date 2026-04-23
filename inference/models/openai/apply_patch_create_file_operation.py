

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ApplyPatchCreateFileOperation(BaseModel):
    """Instruction describing how to create a file via the apply_patch tool."""
    diff: Annotated[str, Field(..., description="Diff to apply.")]
    """Diff to apply."""
    path: Annotated[str, Field(..., description="Path of the file to create.")]
    """Path of the file to create."""
    type: Annotated[Literal["create_file"], Field(default='create_file', description="Create a new file with the provided diff.")]
    """Create a new file with the provided diff."""

    model_config = ConfigDict(extra="ignore")