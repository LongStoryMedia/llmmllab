

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat

from .text_content_block import TextContentBlock
from .tool_use_content_block import ToolUseContentBlock
from .thinking_content_block import ThinkingContentBlock
from .redacted_thinking_content_block import RedactedThinkingContentBlock

OutputContentBlock = Union[
    TextContentBlock,
    ToolUseContentBlock,
    ThinkingContentBlock,
    RedactedThinkingContentBlock,
]
