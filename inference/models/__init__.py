# Auto-generated model exports
# This file was automatically generated to export all models for easy importing

# Import all model modules
try:
    from . import auth_config
    from . import chat_req
    from . import chat_response
    from . import config
    from . import conversation
    from . import database_config
    from . import default_configs
    from . import default_model_profiles
    from . import dev_stats
    from . import image_generation_config
    from . import image_generation_request
    from . import image_generation_response
    from . import image_metadata
    from . import inference_queue_message
    from . import inference_service
    from . import inference_service_config
    from . import internal_config
    from . import lora_weight
    from . import memory
    from . import memory_config
    from . import memory_fragment
    from . import memory_source
    from . import message
    from . import message_content
    from . import message_content_type
    from . import message_role
    from . import model
    from . import model_details
    from . import model_parameters
    from . import model_profile
    from . import model_profile_config
    from . import model_profile_image_settings
    from . import model_task
    from . import preferences_config
    from . import rabbitmq_config
    from . import redis_config
    from . import refinement_config
    from . import requests
    from . import search_result
    from . import search_result_content
    from . import server_config
    from . import summarization_config
    from . import summary
    from . import user
    from . import user_config
    from . import web_search_config
except ImportError as e:
    import sys
    print(f"Warning: Some model modules could not be imported: {e}", file=sys.stderr)

# Define what gets imported with 'from models import *'
__all__ = [
    'auth_config',
    'chat_req',
    'chat_response',
    'config',
    'conversation',
    'database_config',
    'default_configs',
    'default_model_profiles',
    'dev_stats',
    'image_generation_config',
    'image_generation_request',
    'image_generation_response',
    'image_metadata',
    'inference_queue_message',
    'inference_service',
    'inference_service_config',
    'internal_config',
    'lora_weight',
    'memory',
    'memory_config',
    'memory_fragment',
    'memory_source',
    'message',
    'message_content',
    'message_content_type',
    'message_role',
    'model',
    'model_details',
    'model_parameters',
    'model_profile',
    'model_profile_config',
    'model_profile_image_settings',
    'model_task',
    'preferences_config',
    'rabbitmq_config',
    'redis_config',
    'refinement_config',
    'requests',
    'search_result',
    'search_result_content',
    'server_config',
    'summarization_config',
    'summary',
    'user',
    'user_config',
    'web_search_config',
    'AuthConfig',
    'ChatReq',
    'ChatResponse',
    'Config',
    'Conversation',
    'DatabaseConfig',
    'DevStats',
    'ImageGenerationConfig',
    'ImageGenerateRequest',
    'ImageGenerateResponse',
    'ImageMetadata',
    'InferenceQueueMessage',
    'InferenceService',
    'InferenceServiceConfig',
    'InternalConfig',
    'LoraWeight',
    'Memory',
    'MemoryConfig',
    'MemoryFragment',
    'MemorySource',
    'Message',
    'MessageContent',
    'MessageContentType',
    'MessageRole',
    'Model',
    'ModelDetails',
    'ModelParameters',
    'ModelProfile',
    'ModelProfileConfig',
    'ModelProfileImageSettings',
    'ModelTask',
    'PreferencesConfig',
    'RabbitmqConfig',
    'RedisConfig',
    'RefinementConfig',
    'LoraListResponse',
    'LoraWeightRequest',
    'Malloc',
    'ModelRequest',
    'ModelsListResponse',
    'PromptRequest',
    'SearchResult',
    'SearchResultContent',
    'ServerConfig',
    'SummarizationConfig',
    'Summary',
    'User',
    'UserConfig',
    'WebSearchConfig',
]

# Re-export all model classes for easy importing and IDE autocompletion
from .auth_config import (
    AuthConfig,
)
from .chat_req import (
    ChatReq,
)
from .chat_response import (
    ChatResponse,
)
from .config import (
    Config,
)
from .conversation import (
    Conversation,
)
from .database_config import (
    DatabaseConfig,
)
from .dev_stats import (
    DevStats,
)
from .image_generation_config import (
    ImageGenerationConfig,
)
from .image_generation_request import (
    ImageGenerateRequest,
)
from .image_generation_response import (
    ImageGenerateResponse,
)
from .image_metadata import (
    ImageMetadata,
)
from .inference_queue_message import (
    InferenceQueueMessage,
)
from .inference_service import (
    InferenceService,
)
from .inference_service_config import (
    InferenceServiceConfig,
)
from .internal_config import (
    InternalConfig,
)
from .lora_weight import (
    LoraWeight,
)
from .memory import (
    Memory,
)
from .memory_config import (
    MemoryConfig,
)
from .memory_fragment import (
    MemoryFragment,
)
from .memory_source import (
    MemorySource,
)
from .message import (
    Message,
)
from .message_content import (
    MessageContent,
)
from .message_content_type import (
    MessageContentType,
)
from .message_role import (
    MessageRole,
)
from .model import (
    Model,
)
from .model_details import (
    ModelDetails,
)
from .model_parameters import (
    ModelParameters,
)
from .model_profile import (
    ModelProfile,
)
from .model_profile_config import (
    ModelProfileConfig,
)
from .model_profile_image_settings import (
    ModelProfileImageSettings,
)
from .model_task import (
    ModelTask,
)
from .preferences_config import (
    PreferencesConfig,
)
from .rabbitmq_config import (
    RabbitmqConfig,
)
from .redis_config import (
    RedisConfig,
)
from .refinement_config import (
    RefinementConfig,
)
from .requests import (
    LoraListResponse,
    LoraWeightRequest,
    Malloc,
    ModelRequest,
    ModelsListResponse,
    PromptRequest,
)
from .search_result import (
    SearchResult,
)
from .search_result_content import (
    SearchResultContent,
)
from .server_config import (
    ServerConfig,
)
from .summarization_config import (
    SummarizationConfig,
)
from .summary import (
    Summary,
)
from .user import (
    User,
)
from .user_config import (
    UserConfig,
)
from .web_search_config import (
    WebSearchConfig,
)