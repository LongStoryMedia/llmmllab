

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, constr



class MessageType(str, Enum):
    """Type of message"""
    CONNECT = 'connect'
    CONNECTED = 'connected'
    PAUSE = 'pause'
    PAUSED = 'paused'
    RESUME = 'resume'
    RESUMED = 'resumed'
    CANCEL = 'cancel'
    CANCELLED = 'cancelled'
    COMPLETE = 'complete'
    ERROR = 'error'
    WARNING = 'warning'
    INFO = 'info'
    FAILED = 'failed'