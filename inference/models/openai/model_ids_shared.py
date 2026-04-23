

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .chat_model import ChatModel
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat


ModelIdsShared = Union[ChatModel, str]
