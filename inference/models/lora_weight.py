

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class LoraWeight(BaseModel):
    """LoraWeight contains information about a LoRA weight file"""
    id: Annotated[str, Field(..., description="Unique identifier for the LoRA weight")]
    """Unique identifier for the LoRA weight"""
    name: Annotated[str, Field(..., description="Name of the LoRA weight")]
    """Name of the LoRA weight"""
    parent_model: Annotated[str, Field(..., description="Identifier of the parent model this LoRA weight is associated with")]
    """Identifier of the parent model this LoRA weight is associated with"""
    weight_name: Annotated[Optional[str], Field(default=None, description="Name of the weight file (e.g., 'lora.safetensors')")] = None
    """Name of the weight file (e.g., 'lora.safetensors')"""
    adapter_name: Annotated[Optional[str], Field(default=None, description="Name of the adapter (e.g., 'uncensored')")] = None
    """Name of the adapter (e.g., 'uncensored')"""

    model_config = ConfigDict(extra="ignore")