

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ModelParameters(BaseModel):
    """Parameters for configuring a language model"""
    num_ctx: Annotated[Optional[int], Field(default=None, description="Size of the context window")] = None
    """Size of the context window"""
    repeat_last_n: Annotated[Optional[int], Field(default=None, description="Number of tokens to consider for repetition penalties")] = None
    """Number of tokens to consider for repetition penalties"""
    repeat_penalty: Annotated[Optional[float], Field(default=None, description="Penalty for repetitions")] = None
    """Penalty for repetitions"""
    temperature: Annotated[Optional[float], Field(default=None, description="Sampling temperature; higher values produce more creative outputs")] = None
    """Sampling temperature; higher values produce more creative outputs"""
    seed: Annotated[Optional[int], Field(default=None, description="Random seed for reproducibility")] = None
    """Random seed for reproducibility"""
    stop: Annotated[Optional[List[str]], Field(default=None, description="Sequences where the model should stop generating")] = None
    """Sequences where the model should stop generating"""
    num_predict: Annotated[Optional[int], Field(default=None, description="Maximum number of tokens to predict")] = None
    """Maximum number of tokens to predict"""
    top_k: Annotated[Optional[int], Field(default=None, description="Limits next token selection to top K options")] = None
    """Limits next token selection to top K options"""
    top_p: Annotated[Optional[float], Field(default=None, description="Limits next token selection to tokens comprising the top P probability mass (nucleus sampling)")] = None
    """Limits next token selection to tokens comprising the top P probability mass (nucleus sampling)"""
    min_p: Annotated[Optional[float], Field(default=None, description="Minimum probability threshold for token selection")] = None
    """Minimum probability threshold for token selection"""
    think: Annotated[Optional[bool], Field(default=None, description="Whether to enable \"thinking\" mode for the model")] = None
    """Whether to enable \"thinking\" mode for the model"""
    max_tokens: Annotated[Optional[int], Field(default=None, description="Maximum number of tokens to generate in a single response")] = None
    """Maximum number of tokens to generate in a single response"""
    n_parts: Annotated[Optional[int], Field(default=None, description="Number of parts to split the model into. -1 means auto.")] = None
    """Number of parts to split the model into. -1 means auto."""
    batch_size: Annotated[Optional[int], Field(default=None, description="Batch size for processing inputs")] = None
    """Batch size for processing inputs"""
    micro_batch_size: Annotated[Optional[int], Field(default=None, description="Micro batch size for processing inputs")] = None
    """Micro batch size for processing inputs"""
    n_gpu_layers: Annotated[Optional[int], Field(default=None, description="Number of model layers to keep on GPU for performance optimization", ge=-1)] = None
    """Number of model layers to keep on GPU for performance optimization"""
    main_gpu: Annotated[Optional[int], Field(default=-1, description="Main GPU device index (-1 for auto-selection)", ge=-1)] = -1
    """Main GPU device index (-1 for auto-selection)"""
    tensor_split: Annotated[Optional[str], Field(default=None, description="Comma-separated fractions of model to put on each GPU (e.g. \"2,5,5\")")] = None
    """Comma-separated fractions of model to put on each GPU (e.g. \"2,5,5\")"""
    split_mode: Annotated[Optional[Literal["none", "layer", "row"]], Field(default='layer', description="How to split model across devices")] = 'layer'
    """How to split model across devices"""
    n_cpu_moe: Annotated[Optional[int], Field(default=0, description="Number of MoE (Mixture of Experts) layers to keep on CPU for memory optimization", ge=0)] = 0
    """Number of MoE (Mixture of Experts) layers to keep on CPU for memory optimization"""
    kv_on_cpu: Annotated[Optional[bool], Field(default=False, description="Store key-value cache on CPU to save GPU memory")] = False
    """Store key-value cache on CPU to save GPU memory"""
    reasoning_effort: Annotated[Optional[Literal["low", "medium", "high"]], Field(default='medium', description="Reasoning effort level for chain-of-thought processing")] = 'medium'
    """Reasoning effort level for chain-of-thought processing"""
    flash_attention: Annotated[Optional[bool], Field(default=True, description="Enable flash attention optimization for memory efficiency")] = True
    """Enable flash attention optimization for memory efficiency"""

    model_config = ConfigDict(extra="ignore")