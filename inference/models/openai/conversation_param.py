

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .conversation_param_2 import ConversationParam2
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



ConversationParam = Union[ConversationParam2]