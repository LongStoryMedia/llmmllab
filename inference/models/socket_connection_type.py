

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, constr



class SocketConnectionType(str, Enum):
    """SocketConnectionType represents the type of connection used in a WebSocket communication"""
    IMAGE = 'image'
    CHAT = 'chat'
    STATUS = 'status'