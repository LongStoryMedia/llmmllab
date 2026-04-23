

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, constr



class ResponseFormat(str, Enum):
    """Format options for engineering responses"""
    DETAILED_ANALYSIS = 'detailed_analysis'
    CODE_SOLUTION = 'code_solution'
    STEP_BY_STEP_GUIDE = 'step_by_step_guide'
    BEST_PRACTICES = 'best_practices'
    TROUBLESHOOTING = 'troubleshooting'