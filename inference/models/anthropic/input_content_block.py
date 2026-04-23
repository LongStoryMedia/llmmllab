

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .cache_control import CacheControl
from .document_content_block import DocumentContentBlock
from .document_source import DocumentSource
from .image_content_block import ImageContentBlock
from .image_source import ImageSource
from .text_content_block import TextContentBlock
from .tool_reference_content_block import ToolReferenceContentBlock
from .tool_result_content_block import ToolResultContentBlock
from .tool_use_content_block import ToolUseContentBlock
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



InputContentBlock = Union[TextContentBlock, ImageContentBlock, DocumentContentBlock, ToolUseContentBlock, ToolResultContentBlock]