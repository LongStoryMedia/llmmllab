

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ImageGenerateRequest(BaseModel):
    """Request for generating an image"""
    prompt: Annotated[str, Field(..., description="The prompt for image generation")]
    """The prompt for image generation"""
    width: Annotated[Optional[int], Field(default=None, description="Width of the generated image")] = None
    """Width of the generated image"""
    height: Annotated[Optional[int], Field(default=None, description="Height of the generated image")] = None
    """Height of the generated image"""
    inference_steps: Annotated[Optional[int], Field(default=None, description="Number of inference steps")] = None
    """Number of inference steps"""
    guidance_scale: Annotated[Optional[float], Field(default=None, description="Guidance scale for image generation")] = None
    """Guidance scale for image generation"""
    low_memory_mode: Annotated[Optional[bool], Field(default=None, description="Whether to use low memory mode")] = None
    """Whether to use low memory mode"""
    negative_prompt: Annotated[Optional[str], Field(default=None, description="Negative prompt for image generation")] = None
    """Negative prompt for image generation"""
    model: Annotated[str, Field(..., description="Model to use for image generation")]
    """Model to use for image generation"""
    loras: Annotated[Optional[List[str]], Field(default=None, description="List of LoRA models to apply to the image generation")] = None
    """List of LoRA models to apply to the image generation"""
    image_id: Annotated[Optional[int], Field(default=None, description="ID of the image to use as a base for image generation (for img2img)")] = None
    """ID of the image to use as a base for image generation (for img2img)"""
    url: Annotated[Optional[str], Field(default=None, description="URL of the image to use as a base for image generation (for img2img)")] = None
    """URL of the image to use as a base for image generation (for img2img)"""
    filename: Annotated[Optional[str], Field(default=None, description="Name of the file to save the generated image as")] = None
    """Name of the file to save the generated image as"""

    model_config = ConfigDict(extra="ignore")