

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat

from .client_tool import ClientTool
from .server_tool import ServerTool

Tool = Union[ClientTool, ServerTool]
