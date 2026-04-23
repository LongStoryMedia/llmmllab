

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, constr



class MessageRole(str, Enum):
    """Role of a participant in a chat conversation"""
    USER = 'user'
    ASSISTANT = 'assistant'
    SYSTEM = 'system'
    TOOL = 'tool'
    AGENT = 'agent'
    OBSERVER = 'observer'