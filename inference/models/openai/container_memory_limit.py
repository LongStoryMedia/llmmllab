

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, constr



class ContainerMemoryLimit(str, Enum):
    _1G = '1g'
    _4G = '4g'
    _16G = '16g'
    _64G = '64g'