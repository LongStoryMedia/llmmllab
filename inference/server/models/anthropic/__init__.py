# Auto-generated model exports
# This file was automatically generated to export all models for easy importing

from __future__ import annotations

# Suppress Pydantic warnings about fields shadowing BaseModel attributes
# (e.g. 'schema' field in OpenAI models shadows deprecated BaseModel.schema())
import warnings

warnings.filterwarnings("ignore", message=".*shadows an attribute in parent.*")

# Import all model modules
try:
    from . import batch_request
    from . import batch_request_count
    from . import batch_response
    from . import cache_control
    from . import client_tool
    from . import completion_response
    from . import count_tokens_request
    from . import count_tokens_response
    from . import create_batch_request
    from . import create_completion_request
    from . import create_message_request
    from . import delete_response
    from . import document_content_block
    from . import document_source
    from . import error_detail
    from . import error_response
    from . import file_list_response
    from . import file_metadata
    from . import image_content_block
    from . import image_source
    from . import input_content_block
    from . import input_message
    from . import input_schema
    from . import message_response
    from . import metadata
    from . import model
    from . import model_list_response
    from . import output_content_block
    from . import redacted_thinking_content_block
    from . import server_tool
    from . import system_prompt
    from . import text_content_block
    from . import thinking_config
    from . import thinking_content_block
    from . import tool
    from . import tool_choice
    from . import tool_result_content_block
    from . import tool_use_content_block
    from . import usage
except ImportError as e:
    import sys

    print(f"Warning: Some model modules could not be imported: {e}", file=sys.stderr)

# Define what gets imported with 'from models import *'
__all__ = [
    "batch_request",
    "batch_request_count",
    "batch_response",
    "cache_control",
    "client_tool",
    "completion_response",
    "count_tokens_request",
    "count_tokens_response",
    "create_batch_request",
    "create_completion_request",
    "create_message_request",
    "delete_response",
    "document_content_block",
    "document_source",
    "error_detail",
    "error_response",
    "file_list_response",
    "file_metadata",
    "image_content_block",
    "image_source",
    "input_content_block",
    "input_message",
    "input_schema",
    "message_response",
    "metadata",
    "model",
    "model_list_response",
    "output_content_block",
    "redacted_thinking_content_block",
    "server_tool",
    "system_prompt",
    "text_content_block",
    "thinking_config",
    "thinking_content_block",
    "tool",
    "tool_choice",
    "tool_result_content_block",
    "tool_use_content_block",
    "usage",
    "BatchRequest",
    "BatchRequestCount",
    "BatchRequestCount",
    "BatchResponse",
    "CacheControl",
    "ClientTool",
    "CompletionResponse",
    "CountTokensRequest",
    "CountTokensResponse",
    "CreateBatchRequest",
    "CreateCompletionRequest",
    "CreateMessageRequest",
    "DeleteResponse",
    "DocumentContentBlock",
    "DocumentSource",
    "ErrorDetail",
    "ErrorResponse",
    "FileListResponse",
    "FileMetadata",
    "ImageContentBlock",
    "ImageSource",
    "InputMessage",
    "InputSchema",
    "MessageResponse",
    "Metadata",
    "Model",
    "ModelListResponse",
    "RedactedThinkingContentBlock",
    "ServerTool",
    "TextContentBlock",
    "ThinkingConfig",
    "ThinkingContentBlock",
    "CacheControl",
    "ClientTool",
    "InputSchema",
    "ServerTool",
    "ToolChoice",
    "Tool",
    "CacheControl",
    "ImageContentBlock",
    "ImageSource",
    "TextContentBlock",
    "ToolResultContentBlock",
    "ToolUseContentBlock",
    "ServerToolUse",
    "Usage",
]

# Re-export all model classes for easy importing and IDE autocompletion
from .batch_request import (
    BatchRequest,
)
from .batch_request_count import (
    BatchRequestCount,
)
from .batch_response import (
    BatchResponse,
)
from .cache_control import (
    CacheControl,
)
from .client_tool import (
    ClientTool,
)
from .completion_response import (
    CompletionResponse,
)
from .count_tokens_request import (
    CountTokensRequest,
)
from .count_tokens_response import (
    CountTokensResponse,
)
from .create_batch_request import (
    CreateBatchRequest,
)
from .create_completion_request import (
    CreateCompletionRequest,
)
from .create_message_request import (
    CreateMessageRequest,
)
from .delete_response import (
    DeleteResponse,
)
from .document_content_block import (
    DocumentContentBlock,
)
from .document_source import (
    DocumentSource,
)
from .error_detail import (
    ErrorDetail,
)
from .error_response import (
    ErrorResponse,
)
from .file_list_response import (
    FileListResponse,
)
from .file_metadata import (
    FileMetadata,
)
from .image_content_block import (
    ImageContentBlock,
)
from .image_source import (
    ImageSource,
)
from .input_message import (
    InputMessage,
)
from .input_schema import (
    InputSchema,
)
from .message_response import (
    MessageResponse,
)
from .metadata import (
    Metadata,
)
from .model import (
    Model,
)
from .model_list_response import (
    ModelListResponse,
)
from .redacted_thinking_content_block import (
    RedactedThinkingContentBlock,
)
from .server_tool import (
    ServerTool,
)
from .text_content_block import (
    TextContentBlock,
)
from .thinking_config import (
    ThinkingConfig,
)
from .thinking_content_block import (
    ThinkingContentBlock,
)
from .tool import (
    Tool,
)
from .tool_choice import (
    ToolChoice,
)
from .tool_result_content_block import (
    ToolResultContentBlock,
)
from .tool_use_content_block import (
    ToolUseContentBlock,
)
from .usage import (
    ServerToolUse,
    Usage,
)
