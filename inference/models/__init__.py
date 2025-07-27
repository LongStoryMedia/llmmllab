# Auto-generated model exports
# This file was automatically generated to export all models for easy importing

# Import all model modules
try:
    from . import chat_req
    from . import chat_response
    from . import configs
    from . import conversation
    from . import defaults
    from . import dev_stats
    from . import image_generation_request
    from . import image_generation_response
    from . import inference_queue_message
    from . import lora_weight
    from . import message
    from . import message_content
    from . import message_content_type
    from . import message_role
    from . import model
    from . import model_details
    from . import model_parameters
    from . import model_task
    from . import rabbitmq_config
    from . import requests
except ImportError as e:
    import sys
    print(f"Warning: Some model modules could not be imported: {e}", file=sys.stderr)

# Define what gets imported with 'from models import *'
__all__ = [
    'chat_req',
    'chat_response',
    'configs',
    'conversation',
    'defaults',
    'dev_stats',
    'image_generation_request',
    'image_generation_response',
    'inference_queue_message',
    'lora_weight',
    'message',
    'message_content',
    'message_content_type',
    'message_role',
    'model',
    'model_details',
    'model_parameters',
    'model_task',
    'rabbitmq_config',
    'requests',
    'ChatReq',
    'ChatResponse',
    'Conversation',
    'DevStats',
    'ImageGenerateRequest',
    'ImageGenerateResponse',
    'InferenceQueueMessage',
    'LoraWeight',
    'Message',
    'MessageContent',
    'MessageContentType',
    'MessageRole',
    'Model',
    'ModelDetails',
    'ModelParameters',
    'ModelTask',
    'RabbitmqConfig',
    'LoraListResponse',
    'LoraWeightRequest',
    'Malloc',
    'ModelRequest',
    'ModelsListResponse',
    'PromptRequest',
]

# Re-export all model classes for easy importing and IDE autocompletion
from .chat_req import (
    ChatReq,
)
from .chat_response import (
    ChatResponse,
)
from .conversation import (
    Conversation,
)
from .dev_stats import (
    DevStats,
)
from .image_generation_request import (
    ImageGenerateRequest,
)
from .image_generation_response import (
    ImageGenerateResponse,
)
from .inference_queue_message import (
    InferenceQueueMessage,
)
from .lora_weight import (
    LoraWeight,
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
from .model_task import (
    ModelTask,
)
from .rabbitmq_config import (
    RabbitmqConfig,
)
from .requests import (
    LoraListResponse,
    LoraWeightRequest,
    Malloc,
    ModelRequest,
    ModelsListResponse,
    PromptRequest,
)