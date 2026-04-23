

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .other_chunking_strategy_response_param import OtherChunkingStrategyResponseParam
from .static_chunking_strategy import StaticChunkingStrategy
from .static_chunking_strategy_response_param import StaticChunkingStrategyResponseParam
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



ChunkingStrategyResponse = Union[StaticChunkingStrategyResponseParam, OtherChunkingStrategyResponseParam]