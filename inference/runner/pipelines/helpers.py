

from typing import Dict, List, Optional
import torch

from models import MessageContent, MessageContentType, Model, MessageRole


def get_dtype(model: Model) -> torch.dtype:
    """
    Factory function to return the appropriate data type string.
    """
    # Safely access the dtype attribute and handle possible None values
    dtype = None
    if hasattr(model, 'details') and model.details is not None:
        if hasattr(model.details, 'dtype') and model.details.dtype is not None:
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
        print(f"WARNING! Unsupported dtype '{dtype}' for model {model.name}. Defaulting to float32.")
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
        print(f"WARNING! Unsupported dtype '{dtype}' for model {model.name}. Defaulting to None.")
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
    if hasattr(content, 'type'):
        content_dict["type"] = get_content_type(content.type)
    if hasattr(content, 'text') and content.text is not None and content.text != "":
        content_dict["text"] = content.text
    # Include URL field for image content
    if hasattr(content, 'url') and content.url is not None and content.url != "":
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
    # elif content == MessageContentType.VIDEO:
    #     return "video"
    # elif content == MessageContentType.AUDIO:
    #     return "audio"
    # elif content == MessageContentType.FILE:
    #     return "file"
    else:
        raise ValueError(f"Unknown message content type: {content}")
