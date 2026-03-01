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
    from . import message_response
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
    'batch_request',
    'batch_request_count',
    'batch_response',
    'cache_control',
    'client_tool',
    'completion_response',
    'count_tokens_request',
    'count_tokens_response',
    'create_batch_request',
    'create_completion_request',
    'create_message_request',
    'delete_response',
    'document_content_block',
    'document_source',
    'error_detail',
    'error_response',
    'file_list_response',
    'file_metadata',
    'image_content_block',
    'image_source',
    'input_content_block',
    'input_message',
    'message_response',
    'model',
    'model_list_response',
    'output_content_block',
    'redacted_thinking_content_block',
    'server_tool',
    'system_prompt',
    'text_content_block',
    'thinking_config',
    'thinking_content_block',
    'tool',
    'tool_choice',
    'tool_result_content_block',
    'tool_use_content_block',
    'usage',
    'BatchRequest',
    'CacheControl',
    'ClientTool',
    'CreateMessageRequest',
    'DocumentContentBlock',
    'DocumentSource',
    'ImageContentBlock',
    'ImageSource',
    'InputMessage',
    'ServerTool',
    'SystemPrompt',
    'TextContentBlock',
    'ThinkingConfig',
    'ToolChoice',
    'ToolResultContentBlock',
    'ToolUseContentBlock',
    'BatchRequestCount',
    'BatchRequestCount',
    'BatchResponse',
    'CacheControl',
    'CacheControl',
    'ClientTool',
    'InputSchema',
    'CompletionResponse',
    'CacheControl',
    'ClientTool',
    'CountTokensRequest',
    'DocumentContentBlock',
    'DocumentSource',
    'ImageContentBlock',
    'ImageSource',
    'InputMessage',
    'ServerTool',
    'SystemPrompt',
    'TextContentBlock',
    'ThinkingConfig',
    'ToolResultContentBlock',
    'ToolUseContentBlock',
    'CountTokensResponse',
    'BatchRequest',
    'CacheControl',
    'ClientTool',
    'CreateBatchRequest',
    'CreateMessageRequest',
    'DocumentContentBlock',
    'DocumentSource',
    'ImageContentBlock',
    'ImageSource',
    'InputMessage',
    'ServerTool',
    'SystemPrompt',
    'TextContentBlock',
    'ThinkingConfig',
    'ToolChoice',
    'ToolResultContentBlock',
    'ToolUseContentBlock',
    'CreateCompletionRequest',
    'Metadata',
    'CacheControl',
    'ClientTool',
    'CreateMessageRequest',
    'DocumentContentBlock',
    'DocumentSource',
    'ImageContentBlock',
    'ImageSource',
    'InputMessage',
    'Metadata',
    'ServerTool',
    'SystemPrompt',
    'TextContentBlock',
    'ThinkingConfig',
    'ToolChoice',
    'ToolResultContentBlock',
    'ToolUseContentBlock',
    'DeleteResponse',
    'CacheControl',
    'DocumentContentBlock',
    'DocumentSource',
    'TextContentBlock',
    'CacheControl',
    'DocumentSource',
    'TextContentBlock',
    'ErrorDetail',
    'ErrorDetail',
    'ErrorResponse',
    'FileListResponse',
    'FileMetadata',
    'FileMetadata',
    'CacheControl',
    'ImageContentBlock',
    'ImageSource',
    'ImageSource',
    'CacheControl',
    'DocumentContentBlock',
    'DocumentSource',
    'ImageContentBlock',
    'ImageSource',
    'TextContentBlock',
    'ToolResultContentBlock',
    'ToolUseContentBlock',
    'CacheControl',
    'DocumentContentBlock',
    'DocumentSource',
    'ImageContentBlock',
    'ImageSource',
    'InputMessage',
    'TextContentBlock',
    'ToolResultContentBlock',
    'ToolUseContentBlock',
    'CacheControl',
    'MessageResponse',
    'RedactedThinkingContentBlock',
    'TextContentBlock',
    'ThinkingContentBlock',
    'ToolUseContentBlock',
    'Usage',
    'Model',
    'Model',
    'ModelListResponse',
    'CacheControl',
    'RedactedThinkingContentBlock',
    'TextContentBlock',
    'ThinkingContentBlock',
    'ToolUseContentBlock',
    'RedactedThinkingContentBlock',
    'ServerTool',
    'CacheControl',
    'SystemPrompt',
    'TextContentBlock',
    'CacheControl',
    'TextContentBlock',
    'ThinkingConfig',
    'ThinkingContentBlock',
    'CacheControl',
    'ClientTool',
    'InputSchema',
    'ServerTool',
    'ToolChoice',
    'CacheControl',
    'ImageContentBlock',
    'ImageSource',
    'TextContentBlock',
    'ToolResultContentBlock',
    'ToolUseContentBlock',
    'ServerToolUse',
    'Usage',
]

# Re-export all model classes for easy importing and IDE autocompletion
from .batch_request import (
    BatchRequest,
    CacheControl,
    ClientTool,
    CreateMessageRequest,
    DocumentContentBlock,
    DocumentSource,
    ImageContentBlock,
    ImageSource,
    InputMessage,
    ServerTool,
    SystemPrompt,
    TextContentBlock,
    ThinkingConfig,
    ToolChoice,
    ToolResultContentBlock,
    ToolUseContentBlock,
)
from .batch_request_count import (
    BatchRequestCount,
)
from .batch_response import (
    BatchRequestCount,
    BatchResponse,
)
from .cache_control import (
    CacheControl,
)
from .client_tool import (
    CacheControl,
    ClientTool,
    InputSchema,
)
from .completion_response import (
    CompletionResponse,
)
from .count_tokens_request import (
    CacheControl,
    ClientTool,
    CountTokensRequest,
    DocumentContentBlock,
    DocumentSource,
    ImageContentBlock,
    ImageSource,
    InputMessage,
    ServerTool,
    SystemPrompt,
    TextContentBlock,
    ThinkingConfig,
    ToolResultContentBlock,
    ToolUseContentBlock,
)
from .count_tokens_response import (
    CountTokensResponse,
)
from .create_batch_request import (
    BatchRequest,
    CacheControl,
    ClientTool,
    CreateBatchRequest,
    CreateMessageRequest,
    DocumentContentBlock,
    DocumentSource,
    ImageContentBlock,
    ImageSource,
    InputMessage,
    ServerTool,
    SystemPrompt,
    TextContentBlock,
    ThinkingConfig,
    ToolChoice,
    ToolResultContentBlock,
    ToolUseContentBlock,
)
from .create_completion_request import (
    CreateCompletionRequest,
    Metadata,
)
from .create_message_request import (
    CacheControl,
    ClientTool,
    CreateMessageRequest,
    DocumentContentBlock,
    DocumentSource,
    ImageContentBlock,
    ImageSource,
    InputMessage,
    Metadata,
    ServerTool,
    SystemPrompt,
    TextContentBlock,
    ThinkingConfig,
    ToolChoice,
    ToolResultContentBlock,
    ToolUseContentBlock,
)
from .delete_response import (
    DeleteResponse,
)
from .document_content_block import (
    CacheControl,
    DocumentContentBlock,
    DocumentSource,
    TextContentBlock,
)
from .document_source import (
    CacheControl,
    DocumentSource,
    TextContentBlock,
)
from .error_detail import (
    ErrorDetail,
)
from .error_response import (
    ErrorDetail,
    ErrorResponse,
)
from .file_list_response import (
    FileListResponse,
    FileMetadata,
)
from .file_metadata import (
    FileMetadata,
)
from .image_content_block import (
    CacheControl,
    ImageContentBlock,
    ImageSource,
)
from .image_source import (
    ImageSource,
)
from .input_content_block import (
    CacheControl,
    DocumentContentBlock,
    DocumentSource,
    ImageContentBlock,
    ImageSource,
    TextContentBlock,
    ToolResultContentBlock,
    ToolUseContentBlock,
)
from .input_message import (
    CacheControl,
    DocumentContentBlock,
    DocumentSource,
    ImageContentBlock,
    ImageSource,
    InputMessage,
    TextContentBlock,
    ToolResultContentBlock,
    ToolUseContentBlock,
)
from .message_response import (
    CacheControl,
    MessageResponse,
    RedactedThinkingContentBlock,
    TextContentBlock,
    ThinkingContentBlock,
    ToolUseContentBlock,
    Usage,
)
from .model import (
    Model,
)
from .model_list_response import (
    Model,
    ModelListResponse,
)
from .output_content_block import (
    CacheControl,
    RedactedThinkingContentBlock,
    TextContentBlock,
    ThinkingContentBlock,
    ToolUseContentBlock,
)
from .redacted_thinking_content_block import (
    RedactedThinkingContentBlock,
)
from .server_tool import (
    ServerTool,
)
from .system_prompt import (
    CacheControl,
    SystemPrompt,
    TextContentBlock,
)
from .text_content_block import (
    CacheControl,
    TextContentBlock,
)
from .thinking_config import (
    ThinkingConfig,
)
from .thinking_content_block import (
    ThinkingContentBlock,
)
from .tool import (
    CacheControl,
    ClientTool,
    InputSchema,
    ServerTool,
)
from .tool_choice import (
    ToolChoice,
)
from .tool_result_content_block import (
    CacheControl,
    ImageContentBlock,
    ImageSource,
    TextContentBlock,
    ToolResultContentBlock,
)
from .tool_use_content_block import (
    ToolUseContentBlock,
)
from .usage import (
    ServerToolUse,
    Usage,
)