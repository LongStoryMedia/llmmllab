

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .lora_weight import LoraWeight
from .model_details import ModelDetails
from .model_parameters import ModelParameters
from .model_profile_image_settings import ModelProfileImageSettings
from .model_provider import ModelProvider
from .model_task import ModelTask
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class Model(BaseModel):
    """Model represents a machine learning model used for generating responses"""
    id: Annotated[Optional[str], Field(default=None, description="Unique identifier for the model")] = None
    """Unique identifier for the model"""
    name: Annotated[str, Field(..., description="Display name of the model")]
    """Display name of the model"""
    model: Annotated[str, Field(..., description="Identifier used to reference the model")]
    """Identifier used to reference the model"""
    task: Annotated[ModelTask, Field(..., description="Type of task the model is designed for (e.g., \"TextToText\", \"TextToImage\")")]
    """Type of task the model is designed for (e.g., \"TextToText\", \"TextToImage\")"""
    modified_at: Annotated[str, Field(..., description="Timestamp of when the model was last modified")]
    """Timestamp of when the model was last modified"""
    digest: Annotated[str, Field(..., description="Hash digest identifying the model version")]
    """Hash digest identifying the model version"""
    details: Annotated[ModelDetails, Field(..., description="Additional information about the model")]
    """Additional information about the model"""
    pipeline: Annotated[Optional[str], Field(default=None, description="Pipeline type used for the model (e.g., \"sd3\", \"sdxl\")")] = None
    """Pipeline type used for the model (e.g., \"sd3\", \"sdxl\")"""
    lora_weights: Annotated[Optional[List[LoraWeight]], Field(default=None, description="List of LoRA weights associated with the model")] = None
    """List of LoRA weights associated with the model"""
    provider: Annotated[ModelProvider, Field(..., description="Provider or runtime of the model (e.g., \"llama.cpp\", \"hf\", \"openai\")")]
    """Provider or runtime of the model (e.g., \"llama.cpp\", \"hf\", \"openai\")"""
    system_prompt: Annotated[Optional[str], Field(default=None, description="System prompt to use when running this model")] = None
    """System prompt to use when running this model"""
    parameters: Annotated[Optional[ModelParameters], Field(default=None, description="Default inference parameters for this model")] = None
    """Default inference parameters for this model"""
    image_settings: Annotated[Optional[ModelProfileImageSettings], Field(default=None, description="Image generation settings (for image models)")] = None
    """Image generation settings (for image models)"""
    draft_model: Annotated[Optional[str], Field(default=None, description="Optional draft model for speculative decoding")] = None
    """Optional draft model for speculative decoding"""

    model_config = ConfigDict(extra="ignore")