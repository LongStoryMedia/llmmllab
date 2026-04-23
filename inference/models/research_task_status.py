

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, constr



class ResearchTaskStatus(str, Enum):
    """Current status of the research task"""
    PENDING = 'PENDING'
    PLANNING = 'PLANNING'
    GATHERING = 'GATHERING'
    PROCESSING = 'PROCESSING'
    SYNTHESIZING = 'SYNTHESIZING'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'
    CANCELED = 'CANCELED'