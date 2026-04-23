

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, constr



class SummaryType(str, Enum):
    """Types of summaries supported by the SummarizationAgent"""
    PRIMARY = 'primary'
    MASTER = 'master'
    BRIEF = 'brief'
    KEY_POINTS = 'key_points'