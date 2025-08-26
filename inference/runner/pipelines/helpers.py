"""
Helper functions for message processing.
"""

from typing import Dict, Optional, Any
import logging
from llama_cpp import (
    ChatCompletionFunctionParameters,
    ChatCompletionTool,
    ChatCompletionToolFunction,
)
from pydantic import BaseModel
import torch

from langchain_community.tools import BaseTool
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from models import MessageContent, MessageContentType, Model, MessageRole, Message

logger = logging.getLogger(__name__)


def get_dtype(model: Model) -> torch.dtype:
    """
    Factory function to return the appropriate data type string.
    """
    # Safely access the dtype attribute and handle possible None values
    dtype = None
    if hasattr(model, "details") and model.details is not None:
        if hasattr(model.details, "dtype") and model.details.dtype is not None:
            dtype = model.details.dtype.lower()

    # Default to float32 if dtype is None or empty
    if not dtype:
        dtype = "float32"

    if dtype in ["float16", "fp16"]:
        return torch.float16
    elif dtype in ["bfloat16", "bfp16", "bf16"]:
        return torch.bfloat16
    elif dtype in ["float32", "fp32"]:
        return torch.float32
    else:
        print(
            f"WARNING! Unsupported dtype '{dtype}' for model {model.name}. Defaulting to float32."
        )
        return torch.float32


def get_precision(model: Model) -> Optional[str]:
    """
    Factory function to return the appropriate precision string.
    """
    if not model.details or not model.details.dtype:
        return None

    dtype = model.details.dtype.lower()
    if dtype in ["float16", "fp16"]:
        return "fp16"
    elif dtype in ["bfloat16", "bfp16"]:
        return "bfp16"
    elif dtype in ["float32", "fp32"]:
        return "fp32"
    else:
        print(
            f"WARNING! Unsupported dtype '{dtype}' for model {model.name}. Defaulting to None."
        )
        return None


def get_role(role: MessageRole) -> str:
    """
    Convert protobuf MessageRole to string representation.
    """
    if role == MessageRole.USER:
        return "user"
    elif role == MessageRole.ASSISTANT:
        return "assistant"
    elif role == MessageRole.SYSTEM:
        return "system"
    elif role == MessageRole.AGENT:
        return "agent"
    elif role == MessageRole.TOOL:
        return "tool"
    elif role == MessageRole.OBSERVER:
        return "observer"
    else:
        raise ValueError(f"Unknown message role: {role}")


def get_content(content: MessageContent) -> Dict[str, str]:
    """
    Convert protobuf MessageContent to string representation.
    """

    content_dict = {}
    if hasattr(content, "type"):
        content_dict["type"] = get_content_type(content.type)
    if hasattr(content, "text") and content.text is not None and content.text != "":
        content_dict["text"] = content.text
    # Include URL field for image content
    if hasattr(content, "url") and content.url is not None and content.url != "":
        content_dict["url"] = content.url

    return content_dict


def get_content_type(content: MessageContentType) -> str:
    """
    Get the content type from MessageContent.
    """
    if content == MessageContentType.TEXT:
        return "text"
    elif content == MessageContentType.IMAGE:
        return "image"
    elif content == MessageContentType.VIDEO:
        return "video"
    elif content == MessageContentType.AUDIO:
        return "audio"
    elif content == MessageContentType.FILE:
        return "file"
    else:
        raise ValueError(f"Unknown message content type: {content}")


def to_lc_message(message: Message) -> BaseMessage:
    """Convert a Message object to a format suitable for LangChain."""
    # Extract text content as a simple string
    text_content = extract_message_text(message)

    if message.role == MessageRole.ASSISTANT:
        return AIMessage(content=text_content)
    elif message.role == MessageRole.USER:
        return HumanMessage(content=text_content)
    elif message.role == MessageRole.SYSTEM:
        return SystemMessage(content=text_content)
    else:
        # Default to HumanMessage for unknown roles to ensure we always return a BaseMessage
        logger.warning(
            f"Unknown message role: {message.role}, defaulting to HumanMessage"
        )
        return HumanMessage(content=text_content)


def from_lc_message(lc_message: BaseMessage) -> Message:
    """Convert a LangChain message to a Message object."""
    if isinstance(lc_message, AIMessage):
        role = MessageRole.ASSISTANT
    elif isinstance(lc_message, HumanMessage):
        role = MessageRole.USER
    elif isinstance(lc_message, SystemMessage):
        role = MessageRole.SYSTEM
    else:
        logger.warning(
            f"Unknown LangChain message type: {type(lc_message)}, defaulting to USER"
        )
        role = MessageRole.USER

    # Convert content to string regardless of type
    if isinstance(lc_message.content, str):
        text_content = lc_message.content
    elif isinstance(lc_message.content, list):
        # Join list items or convert them to string
        text_content = str(lc_message.content)
    else:
        # Handle other types by converting to string
        text_content = str(lc_message.content) if lc_message.content else ""

    return Message(
        role=role,
        content=[
            MessageContent(
                type=MessageContentType.TEXT,
                text=text_content,
                url=None,
            )
        ],
    )


def extract_message_text(message: Message) -> str:
    """Extract text content from a message object"""
    text_parts = []
    for content in message.content:
        if content.type == MessageContentType.TEXT and content.text:
            text_parts.append(content.text)
    return "\n".join(text_parts).strip()


def lc_to_llama_tool(lc_tool: BaseTool) -> Optional[ChatCompletionTool]:
    """
    Convert a LangChain BaseTool to llama-cpp-python ChatCompletionTool format.

    Args:
        tool: LangChain BaseTool instance

    Returns:
        Dictionary in ChatCompletionTool format for llama-cpp-python
    """
    # Get the tool's input schema
    if hasattr(lc_tool, "args_schema") and lc_tool.args_schema:
        # Get schema from Pydantic model
        if isinstance(lc_tool.args_schema, dict):
            schema = lc_tool.args_schema
        else:
            schema = lc_tool.args_schema.model_json_schema()
    else:
        # Fallback to empty schema if no args_schema
        schema = {"type": "object", "properties": {}}

    # Build the ChatCompletionTool structure
    chat_completion_tool = ChatCompletionTool(
        type="function",
        function=ChatCompletionToolFunction(
            name=lc_tool.name,
            description=lc_tool.description or f"Execute {lc_tool.name}",
            parameters=convert_pydantic_schema_to_json_schema(schema),
        ),
    )

    return chat_completion_tool


def convert_pydantic_schema_to_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Pydantic schema to JSON Schema format compatible with OpenAI function calling.
    """
    if not schema:
        return {"type": "object", "properties": {}}

    # Handle the case where schema is already in the correct format
    if "properties" in schema and "type" in schema:
        return schema

    # Convert from Pydantic schema format
    json_schema = {"type": "object", "properties": {}}

    if "properties" in schema:
        json_schema["properties"] = schema["properties"]

    if "required" in schema:
        json_schema["required"] = schema["required"]

    if "description" in schema:
        json_schema["description"] = schema["description"]

    return json_schema
