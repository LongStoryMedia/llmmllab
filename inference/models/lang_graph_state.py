

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .document import Document
from .message import Message
from .message_content import MessageContent
from .message_content_type import MessageContentType
from .message_role import MessageRole
from .resource_usage import ResourceUsage
from .thought import Thought
from .tool_call import ToolCall
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class LangGraphState(BaseModel):
    """Strongly typed state for LangGraph workflows."""
    messages: Annotated[List[Message], Field(..., description="Sequence of messages")]
    """Sequence of messages"""
    user_input: Annotated[str, Field(..., description="Latest user input")]
    """Latest user input"""
    error_count: Annotated[int, Field(default=0, description="Number of errors encountered during workflow execution")]
    """Number of errors encountered during workflow execution"""
    max_iterations: Annotated[int, Field(default=10, description="Maximum number of iterations allowed in the workflow")]
    """Maximum number of iterations allowed in the workflow"""
    current_iteration: Annotated[int, Field(default=0, description="Current iteration count in the workflow execution")]
    """Current iteration count in the workflow execution"""
    tools_used: Annotated[List[str], Field(default_factory=lambda: [], description="List of tools or functions that have been used in the workflow")]
    """List of tools or functions that have been used in the workflow"""
    intermediate_results: Annotated[Dict[str, Any], Field(default_factory=lambda: {}, description="Storage for intermediate computation results during workflow execution")]
    """Storage for intermediate computation results during workflow execution"""

    model_config = ConfigDict(extra="ignore")