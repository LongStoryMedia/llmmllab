

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .model_parameters import ModelParameters
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class GenerateReq(BaseModel):
    """GenerateReq represents a request to the Ollama embeddings API"""
    model: Annotated[str, Field(..., description="Model name")]
    """Model name"""
    prompt: Annotated[str, Field(..., description="Prompt to send to the model")]
    """Prompt to send to the model"""
    suffix: Annotated[Optional[str], Field(default=None, description="Suffix to append to the prompt")] = None
    """Suffix to append to the prompt"""
    images: Annotated[Optional[List[str]], Field(default=None, description="Images to include with the prompt")] = None
    """Images to include with the prompt"""
    format: Annotated[Optional[Dict[str, Any]], Field(default=None, description="The format to return a response in. Format can be json or a JSON schema")] = None
    """The format to return a response in. Format can be json or a JSON schema"""
    options: Annotated[Optional[ModelParameters], Field(default=None, description="Additional model parameters listed in the documentation for the Modelfile such as temperature")] = None
    """Additional model parameters listed in the documentation for the Modelfile such as temperature"""
    system: Annotated[Optional[str], Field(default=None, description="System message to (overrides what is defined in the Modelfile)")] = None
    """System message to (overrides what is defined in the Modelfile)"""
    template: Annotated[Optional[str], Field(default=None, description="The prompt template to use (overrides what is defined in the Modelfile)")] = None
    """The prompt template to use (overrides what is defined in the Modelfile)"""
    stream: Annotated[Optional[bool], Field(default=None, description="If false the response will be returned as a single response object")] = None
    """If false the response will be returned as a single response object"""
    raw: Annotated[Optional[bool], Field(default=None, description="If true no formatting will be applied to the prompt")] = None
    """If true no formatting will be applied to the prompt"""
    keep_alive: Annotated[Optional[int], Field(default=None, description="Controls how long the model will stay loaded into memory")] = None
    """Controls how long the model will stay loaded into memory"""
    context: Annotated[Optional[str], Field(default=None, description="the context parameter returned from a previous request")] = None
    """the context parameter returned from a previous request"""
    think: Annotated[Optional[bool], Field(default=None, description="If true, the model will think before responding, useful for complex queries")] = None
    """If true, the model will think before responding, useful for complex queries"""

    model_config = ConfigDict(extra="ignore")