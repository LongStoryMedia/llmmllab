

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .cache_control import CacheControl
from .document_source import DocumentSource
from .text_content_block import TextContentBlock
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class DocumentContentBlock(BaseModel):
    type: Annotated[Literal["document"], Field(...)]
    source: Annotated[DocumentSource, Field(...)]
    title: Annotated[Optional[str], Field(default=None, description="Optional title for the document.")] = None
    """Optional title for the document."""
    context: Annotated[Optional[str], Field(default=None, description="Optional context/description for the document.")] = None
    """Optional context/description for the document."""
    cache_control: Annotated[Optional[CacheControl], Field(default=None)] = None

    model_config = ConfigDict(extra="ignore")