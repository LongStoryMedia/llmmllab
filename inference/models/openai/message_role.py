

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, constr



class MessageRole(str, Enum):
    UNKNOWN = 'unknown'
    USER = 'user'
    ASSISTANT = 'assistant'
    SYSTEM = 'system'
    CRITIC = 'critic'
    DISCRIMINATOR = 'discriminator'
    DEVELOPER = 'developer'
    TOOL = 'tool'