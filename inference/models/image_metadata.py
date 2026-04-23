

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ImageMetadata(BaseModel):
    """Metadata for an image generated on the system."""
    id: Annotated[Optional[int], Field(default=None, description="Unique identifier for the image.")] = None
    """Unique identifier for the image."""
    filename: Annotated[str, Field(..., description="filename (including path) of the image.")]
    """filename (including path) of the image."""
    thumbnail: Annotated[str, Field(..., description="URL or path to the thumbnail version of the image.")]
    """URL or path to the thumbnail version of the image."""
    width: Annotated[int, Field(..., description="Width of the image in pixels.")]
    """Width of the image in pixels."""
    height: Annotated[int, Field(..., description="Height of the image in pixels.")]
    """Height of the image in pixels."""
    format: Annotated[Literal["jpg", "png", "gif", "bmp", "webp", "svg"], Field(..., description="Format of the image file.")]
    """Format of the image file."""
    user_id: Annotated[str, Field(..., description="Unique identifier for the user who the image was generated for.")]
    """Unique identifier for the user who the image was generated for."""
    conversation_id: Annotated[Optional[int], Field(default=None, description="Unique identifier for the conversation associated with the image.")] = None
    """Unique identifier for the conversation associated with the image."""
    created_at: Annotated[datetime, Field(..., description="Timestamp when the image was created.")]
    """Timestamp when the image was created."""
    view_url: Annotated[Optional[str], Field(default=None, description="URL for viewing the image.")] = None
    """URL for viewing the image."""
    download_url: Annotated[Optional[str], Field(default=None, description="URL for downloading the image.")] = None
    """URL for downloading the image."""

    model_config = ConfigDict(extra="ignore")