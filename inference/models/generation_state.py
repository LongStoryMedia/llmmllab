

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, constr



class GenerationState(str, Enum):
    """Current state of the generation process."""
    ANALYZING = 'analyzing'
    THINKING = 'thinking'
    EXECUTING = 'executing'
    RESPONDING = 'responding'
    FORMATTING = 'formatting'