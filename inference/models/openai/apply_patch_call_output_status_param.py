

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, constr



class ApplyPatchCallOutputStatusParam(str, Enum):
    """Outcome values reported for apply_patch tool call outputs."""
    COMPLETED = 'completed'
    FAILED = 'failed'