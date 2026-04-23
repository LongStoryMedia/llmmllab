

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ModelDetails(BaseModel):
    """ModelDetails contains additional information about a model"""
    parent_model: Annotated[Optional[str], Field(default=None, description="Identifier of the parent model if this is a derivative")] = None
    """Identifier of the parent model if this is a derivative"""
    format: Annotated[str, Field(..., description="Format of the model file (e.g., gguf)")]
    """Format of the model file (e.g., gguf)"""
    gguf_file: Annotated[Optional[str], Field(default=None, description="Path to the GGUF file if applicable")] = None
    """Path to the GGUF file if applicable"""
    clip_model_path: Annotated[Optional[str], Field(default=None, description="Path to the CLIP model file for multimodal models (e.g., mmproj file)")] = None
    """Path to the CLIP model file for multimodal models (e.g., mmproj file)"""
    family: Annotated[str, Field(..., description="Primary model family this belongs to (e.g., llama, phi3)")]
    """Primary model family this belongs to (e.g., llama, phi3)"""
    families: Annotated[List[str], Field(..., description="All model families this belongs to")]
    """All model families this belongs to"""
    parameter_size: Annotated[str, Field(..., description="Size of model parameters (e.g., '7.2B')")]
    """Size of model parameters (e.g., '7.2B')"""
    quantization_level: Annotated[Optional[str], Field(default=None, description="Level of quantization applied to the model (e.g., 'Q4_0')")] = None
    """Level of quantization applied to the model (e.g., 'Q4_0')"""
    dtype: Annotated[Optional[str], Field(default=None, description="Data type of the model (e.g., 'float16', 'bfloat16')")] = None
    """Data type of the model (e.g., 'float16', 'bfloat16')"""
    precision: Annotated[Optional[Literal["fp32", "fp16", "bf16", "int8", "int4", "int2", "int1"]], Field(default=None, description="Precision of the model (e.g., 'fp16', 'bf16')")] = None
    """Precision of the model (e.g., 'fp16', 'bf16')"""
    specialization: Annotated[Optional[Literal["LoRA", "Embedding", "TextToImage", "ImageToImage", "Audio", "Text"]], Field(default=None, description="Specialization of the model (e.g., 'LoRA', 'Embedding', 'TextToImage', 'Audio', 'Vision')")] = None
    """Specialization of the model (e.g., 'LoRA', 'Embedding', 'TextToImage', 'Audio', 'Vision')"""
    description: Annotated[Optional[str], Field(default=None, description="Description of the model")] = None
    """Description of the model"""
    weight: Annotated[Optional[float], Field(default=None, description="Weight of the model (applies to LoRA models)")] = None
    """Weight of the model (applies to LoRA models)"""
    size: Annotated[int, Field(..., description="Size of the model in bytes")]
    """Size of the model in bytes"""
    original_ctx: Annotated[int, Field(..., description="Original context window size of the model")]
    """Original context window size of the model"""
    n_layers: Annotated[Optional[int], Field(default=None, description="Number of layers in the model")] = None
    """Number of layers in the model"""
    hidden_size: Annotated[Optional[int], Field(default=None, description="Hidden size of the model")] = None
    """Hidden size of the model"""
    n_heads: Annotated[Optional[int], Field(default=None, description="Number of attention heads in the model")] = None
    """Number of attention heads in the model"""
    n_kv_heads: Annotated[Optional[int], Field(default=None, description="Number of key-value heads in the model")] = None
    """Number of key-value heads in the model"""
    clip_model_size: Annotated[Optional[int], Field(default=None, description="Size of the CLIP model in bytes, if applicable")] = None
    """Size of the CLIP model in bytes, if applicable"""

    model_config = ConfigDict(extra="ignore")