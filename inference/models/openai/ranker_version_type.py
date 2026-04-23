

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, constr



class RankerVersionType(str, Enum):
    AUTO = 'auto'
    DEFAULT_2024_11_15 = 'default-2024-11-15'