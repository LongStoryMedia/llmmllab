from typing import List, Optional
from numpy import negative
from pydantic import BaseModel

from .dev_stats import DevStats
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
    max_sequence_length: Optional[int] = None  # Optional max sequence length
    request_id: Optional[str] = None  # Unique request ID for tracking

    def get(self, key: str, default=None):
        """Get a parameter value, falling back to default if not set."""
        return getattr(self, key, default)


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


class Malloc(BaseModel):
    """Response model for RAM usage."""
    devices: dict[str, DevStats]
