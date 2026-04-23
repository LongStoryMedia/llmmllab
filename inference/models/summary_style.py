

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, constr



class SummaryStyle(str, Enum):
    """Styles of summaries supported by the SummarizationAgent"""
    CONCISE = 'concise'
    DETAILED = 'detailed'
    BULLET_POINTS = 'bullet_points'
    STRUCTURED = 'structured'