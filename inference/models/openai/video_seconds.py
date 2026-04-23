

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, constr



class VideoSeconds(str, Enum):
    _4 = '4'
    _8 = '8'
    _12 = '12'