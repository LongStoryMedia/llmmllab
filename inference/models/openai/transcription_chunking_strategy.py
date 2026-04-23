

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .vad_config import VadConfig
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class TranscriptionChunkingStrategy(BaseModel):
    pass