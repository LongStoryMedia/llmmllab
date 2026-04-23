

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ModelProfileImageSettings(BaseModel):
    """ModelProfileImageSettings represents the image settings for a model profile"""
    height: Annotated[Optional[int], Field(default=None, description="Height of the image in pixels")] = None
    """Height of the image in pixels"""
    width: Annotated[Optional[int], Field(default=None, description="Width of the image in pixels")] = None
    """Width of the image in pixels"""
    inference_steps: Annotated[Optional[int], Field(default=None, description="Number of inference steps")] = None
    """Number of inference steps"""
    guidance_scale: Annotated[Optional[float], Field(default=None, description="Guidance scale for image generation")] = None
    """Guidance scale for image generation"""
    low_memory_mode: Annotated[Optional[bool], Field(default=None, description="Whether to use low memory mode")] = None
    """Whether to use low memory mode"""
    negative_prompt: Annotated[Optional[str], Field(default=None, description="Negative prompt for image generation")] = None
    """Negative prompt for image generation"""
    lora_model: Annotated[Optional[str], Field(default=None, description="Name of the LoRA model to use for image generation")] = None
    """Name of the LoRA model to use for image generation"""

    model_config = ConfigDict(extra="ignore")