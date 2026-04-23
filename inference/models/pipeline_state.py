

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, constr



class PipelineState(str, Enum):
    """Pipeline lifecycle states for tracking execution phases"""
    INITIALIZING = 'initializing'
    RUNNING = 'running'
    COMPLETING = 'completing'
    TERMINATED = 'terminated'
    FAILED = 'failed'
    CLEANUP = 'cleanup'