

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class Tool(BaseModel):
    """Schema representing a LangChain BaseTool with its properties and configuration"""
    name: Annotated[str, Field(..., description="The unique name of the tool that clearly communicates its purpose")]
    """The unique name of the tool that clearly communicates its purpose"""
    description: Annotated[str, Field(..., description="Used to tell the model how/when/why to use the tool. You can provide few-shot examples as a part of the description")]
    """Used to tell the model how/when/why to use the tool. You can provide few-shot examples as a part of the description"""
    args_schema: Annotated[Optional[Union[Any, Dict[str, Any], str]], Field(default=None, description="Pydantic model class to validate and parse the tool's input arguments")] = None
    """Pydantic model class to validate and parse the tool's input arguments"""
    return_direct: Annotated[Optional[bool], Field(default=False, description="Whether to return the tool's output directly. Setting this to True means that after the tool is called, the AgentExecutor will stop looping")] = False
    """Whether to return the tool's output directly. Setting this to True means that after the tool is called, the AgentExecutor will stop looping"""
    tags: Annotated[Optional[List[str]], Field(default=None, description="Optional list of tags associated with the tool. These tags will be associated with each call to this tool and passed as arguments to the handlers defined in callbacks")] = None
    """Optional list of tags associated with the tool. These tags will be associated with each call to this tool and passed as arguments to the handlers defined in callbacks"""
    metadata: Annotated[Optional[Dict[str, Any]], Field(default=None, description="Optional metadata associated with the tool. This metadata will be associated with each call to this tool and passed as arguments to the handlers defined in callbacks")] = None
    """Optional metadata associated with the tool. This metadata will be associated with each call to this tool and passed as arguments to the handlers defined in callbacks"""
    handle_tool_error: Annotated[Optional[Union[bool, str, Any]], Field(default=False, description="Handle the content of the ToolException thrown. Can be a boolean, string, or callable function")] = False
    """Handle the content of the ToolException thrown. Can be a boolean, string, or callable function"""
    handle_validation_error: Annotated[Optional[Union[bool, str, Any]], Field(default=False, description="Handle the content of the ValidationError thrown. Can be a boolean, string, or callable function")] = False
    """Handle the content of the ValidationError thrown. Can be a boolean, string, or callable function"""
    response_format: Annotated[Optional[Literal["content", "content_and_artifact"]], Field(default='content', description="The tool response format. If 'content' then the output is interpreted as ToolMessage contents. If 'content_and_artifact' then output is expected to be a two-tuple of (content, artifact)")] = 'content'
    """The tool response format. If 'content' then the output is interpreted as ToolMessage contents. If 'content_and_artifact' then output is expected to be a two-tuple of (content, artifact)"""

    model_config = ConfigDict(extra="ignore")