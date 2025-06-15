from typing import List, Optional
from numpy import negative
from pydantic import BaseModel

from models.model import Model


class PromptRequest(BaseModel):
    prompt: str
    width: Optional[int] = None  # Allow client to specify dimensions
    height: Optional[int] = None
    low_memory_mode: Optional[bool] = False  # Option to use low memory mode
    inference_steps: Optional[int] = 30  # Custom inference steps
    guidance_scale: Optional[float] = 7.5  # Custom guidance scale
    negative_prompt: Optional[str] = None  # Optional negative prompt
    seed: Optional[int] = None  # Optional seed for reproducibility


class ModelRequest(BaseModel):
    """Request model for adding a new model."""
    name: Optional[str] = None
    source: str
    description: Optional[str] = None
    weight: float = 1.0  # Default weight for LoRA models


class ModelsListResponse(BaseModel):
    """Response model for listing models."""
    models: List[Model]
    active_model: str


class LoraListResponse(BaseModel):
    """Response model for listing loras."""
    loras: List[Model]
    active_loras: List[str]


class LoraWeightRequest(BaseModel):
    weight: float
