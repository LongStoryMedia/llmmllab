

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, constr



class SocketStageType(str, Enum):
    """The current processing stage."""
    INITIALIZING = 'initializing'
    RETRIEVING_MEMORIES = 'retrieving_memories'
    SEARCHING_WEB = 'searching_web'
    SUMMARIZING = 'summarizing'
    GENERATING_IMAGE = 'generating_image'
    PROCESSING = 'processing'
    RENDERING = 'rendering'
    INTERPRETING = 'interpreting'
    OPEN = 'open'