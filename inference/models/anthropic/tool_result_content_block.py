

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .cache_control import CacheControl
from .image_content_block import ImageContentBlock
from .image_source import ImageSource
from .text_content_block import TextContentBlock
from .tool_reference_content_block import ToolReferenceContentBlock
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat


class ToolResultContentBlock(BaseModel):
    """Result of a tool call, sent by the user in a subsequent message."""

    type: Annotated[Literal["tool_result"], Field(...)]
    tool_use_id: Annotated[
        str, Field(..., description="The `id` from the corresponding `tool_use` block.")
    ]
    """The `id` from the corresponding `tool_use` block."""
    content: Annotated[
        Optional[
            Union[
                str,
                List[
                    Union[
                        TextContentBlock, ImageContentBlock, ToolReferenceContentBlock
                    ]
                ],
            ]
        ],
        Field(default=None),
    ] = None
    is_error: Annotated[
        Optional[bool],
        Field(default=None, description="Whether this result represents an error."),
    ] = None
    """Whether this result represents an error."""
    cache_control: Annotated[Optional[CacheControl], Field(default=None)] = None

    model_config = ConfigDict(extra="ignore")
