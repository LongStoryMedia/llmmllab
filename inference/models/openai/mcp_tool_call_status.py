

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, constr



class MCPToolCallStatus(str, Enum):
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'
    INCOMPLETE = 'incomplete'
    CALLING = 'calling'
    FAILED = 'failed'