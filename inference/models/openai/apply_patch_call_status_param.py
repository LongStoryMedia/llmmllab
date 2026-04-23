

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, constr



class ApplyPatchCallStatusParam(str, Enum):
    """Status values reported for apply_patch tool calls."""
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'