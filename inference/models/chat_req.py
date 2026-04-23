

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .document import Document
from .message import Message
from .message_content import MessageContent
from .message_content_type import MessageContentType
from .message_role import MessageRole
from .model_parameters import ModelParameters
from .resource_usage import ResourceUsage
from .thought import Thought
from .tool_call import ToolCall
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ChatReq(BaseModel):
    """ChatReq represents a request to the Ollama API"""
    messages: Annotated[List[Message], Field(..., description="Messages to send to the model, each message is a struct with role and content")]
    """Messages to send to the model, each message is a struct with role and content"""
    stream: Annotated[bool, Field(..., description="If true, the response will be streamed back as a series of events")]
    """If true, the response will be streamed back as a series of events"""
    format: Annotated[Optional[Dict[str, Any]], Field(default=None, description="The format to return a response in. Format can be json or a JSON schema")] = None
    """The format to return a response in. Format can be json or a JSON schema"""
    conversation_id: Annotated[int, Field(..., description="The ID of the conversation this message belongs to")]
    """The ID of the conversation this message belongs to"""
    keep_alive: Annotated[Optional[int], Field(default=None, description="Controls how long the model will stay loaded into memory")] = None
    """Controls how long the model will stay loaded into memory"""
    options: Annotated[Optional[ModelParameters], Field(default=None, description="Additional model parameters listed in the documentation for the Modelfile such as temperature")] = None
    """Additional model parameters listed in the documentation for the Modelfile such as temperature"""
    tools: Annotated[Optional[List[Dict[str, Any]]], Field(default=None, description="Tools to use for the request, if any")] = None
    """Tools to use for the request, if any"""
    think: Annotated[Optional[bool], Field(default=None, description="If true, the model will think before responding, useful for complex queries")] = None
    """If true, the model will think before responding, useful for complex queries"""

    model_config = ConfigDict(extra="ignore")