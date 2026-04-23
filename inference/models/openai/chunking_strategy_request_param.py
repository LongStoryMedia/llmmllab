

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .auto_chunking_strategy_request_param import AutoChunkingStrategyRequestParam
from .static_chunking_strategy import StaticChunkingStrategy
from .static_chunking_strategy_request_param import StaticChunkingStrategyRequestParam
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



ChunkingStrategyRequestParam = Union[AutoChunkingStrategyRequestParam, StaticChunkingStrategyRequestParam]