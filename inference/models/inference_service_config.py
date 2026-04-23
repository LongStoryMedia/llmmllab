

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .inference_service import InferenceService
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class InferenceServiceConfig(BaseModel):
    """Inference services configuration"""
    ollama: Annotated[InferenceService, Field(...)]
    stable_diffusion: Annotated[InferenceService, Field(...)]
    host: Annotated[str, Field(..., description="Host for the inference service")]
    """Host for the inference service"""
    port: Annotated[int, Field(..., description="Port for the inference service")]
    """Port for the inference service"""

    model_config = ConfigDict(extra="ignore")