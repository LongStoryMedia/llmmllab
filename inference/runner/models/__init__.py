# Auto-generated model exports for runner
# This file was automatically generated to export all models for easy importing

from __future__ import annotations

# Suppress Pydantic warnings about fields shadowing BaseModel attributes
# (e.g. 'schema' field in OpenAI models shadows deprecated BaseModel.schema())
import warnings
warnings.filterwarnings("ignore", message=".*shadows an attribute in parent.*")

# Import all model modules
try:
    from . import chat_response
    from . import circuit_breaker_config
    from . import complexity_level
    from . import computational_requirement
    from . import crash_prevention
    from . import dev_stats
    from . import document
    from . import event_stream_config
    from . import generation_state
    from . import gpu_config
    from . import image_generation_config
    from . import intent_analysis
    from . import learned_limits
    from . import lora_weight
    from . import memory_config
    from . import message
    from . import message_content
    from . import message_content_type
    from . import message_role
    from . import ml_model_performance
    from . import model
    from . import model_configuration_data
    from . import model_details
    from . import model_parameters
    from . import model_profile
    from . import model_profile_config
    from . import model_profile_image_settings
    from . import model_profile_type
    from . import model_provider
    from . import model_task
    from . import node_metadata
    from . import oom_recovery_attempt_data
    from . import optimal_parameters
    from . import parameter_optimization_config
    from . import parameter_tuning_strategy
    from . import performance_parameter
    from . import pipeline_priority
    from . import prediction_features
    from . import recovery_strategy
    from . import required_capability
    from . import resource_usage
    from . import response_format
    from . import summarization_config
    from . import system_gpu_stats
    from . import technical_domain
    from . import thought
    from . import todo_item
    from . import tool_call
    from . import tool_config
    from . import user_config
    from . import workflow_config
    from . import workflow_type
except ImportError as e:
    import sys
    print(f"Warning: Some model modules could not be imported: {e}", file=sys.stderr)

# Define what gets imported with 'from models import *'
__all__ = [
    'chat_response',
    'circuit_breaker_config',
    'complexity_level',
    'computational_requirement',
    'crash_prevention',
    'dev_stats',
    'document',
    'event_stream_config',
    'generation_state',
    'gpu_config',
    'image_generation_config',
    'intent_analysis',
    'learned_limits',
    'lora_weight',
    'memory_config',
    'message',
    'message_content',
    'message_content_type',
    'message_role',
    'ml_model_performance',
    'model',
    'model_configuration_data',
    'model_details',
    'model_parameters',
    'model_profile',
    'model_profile_config',
    'model_profile_image_settings',
    'model_profile_type',
    'model_provider',
    'model_task',
    'node_metadata',
    'oom_recovery_attempt_data',
    'optimal_parameters',
    'parameter_optimization_config',
    'parameter_tuning_strategy',
    'performance_parameter',
    'pipeline_priority',
    'prediction_features',
    'recovery_strategy',
    'required_capability',
    'resource_usage',
    'response_format',
    'summarization_config',
    'system_gpu_stats',
    'technical_domain',
    'thought',
    'todo_item',
    'tool_call',
    'tool_config',
    'user_config',
    'workflow_config',
    'workflow_type',
    'ChatResponse',
    'CircuitBreakerConfig',
    'ComplexityLevel',
    'ComputationalRequirement',
    'CrashPrevention',
    'DevStats',
    'Document',
    'EventStreamConfig',
    'GenerationState',
    'GPUConfig',
    'ImageGenerationConfig',
    'IntentAnalysis',
    'LearnedLimits',
    'LoraWeight',
    'MemoryConfig',
    'Message',
    'MessageContent',
    'MessageContentType',
    'MessageRole',
    'MLModelPerformance',
    'Model',
    'ModelConfigurationData',
    'ModelDetails',
    'ModelParameters',
    'ModelProfile',
    'ModelProfileConfig',
    'ModelProfileImageSettings',
    'ModelProfileType',
    'ModelProvider',
    'ModelTask',
    'NodeMetadata',
    'OOMRecoveryAttemptData',
    'OptimalParameters',
    'ParameterOptimizationConfig',
    'ParameterTuningStrategy',
    'PerformanceParameter',
    'PipelinePriority',
    'PredictionFeatures',
    'RecoveryStrategy',
    'RequiredCapability',
    'ResourceUsage',
    'ResponseFormat',
    'SummarizationConfig',
    'SystemGPUStats',
    'TechnicalDomain',
    'Thought',
    'TodoItem',
    'ToolCall',
    'ToolConfig',
    'UserConfig',
    'WorkflowConfig',
    'WorkflowType',
]

# Re-export all model classes for easy importing and IDE autocompletion
from .chat_response import (
    ChatResponse,
)
from .circuit_breaker_config import (
    CircuitBreakerConfig,
)
from .complexity_level import (
    ComplexityLevel,
)
from .computational_requirement import (
    ComputationalRequirement,
)
from .crash_prevention import (
    CrashPrevention,
)
from .dev_stats import (
    DevStats,
)
from .document import (
    Document,
)
from .event_stream_config import (
    EventStreamConfig,
)
from .generation_state import (
    GenerationState,
)
from .gpu_config import (
    GPUConfig,
)
from .image_generation_config import (
    ImageGenerationConfig,
)
from .intent_analysis import (
    IntentAnalysis,
)
from .learned_limits import (
    LearnedLimits,
)
from .lora_weight import (
    LoraWeight,
)
from .memory_config import (
    MemoryConfig,
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
from .ml_model_performance import (
    MLModelPerformance,
)
from .model import (
    Model,
)
from .model_configuration_data import (
    ModelConfigurationData,
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
from .model_profile_type import (
    ModelProfileType,
)
from .model_provider import (
    ModelProvider,
)
from .model_task import (
    ModelTask,
)
from .node_metadata import (
    NodeMetadata,
)
from .oom_recovery_attempt_data import (
    OOMRecoveryAttemptData,
)
from .optimal_parameters import (
    OptimalParameters,
)
from .parameter_optimization_config import (
    ParameterOptimizationConfig,
)
from .parameter_tuning_strategy import (
    ParameterTuningStrategy,
)
from .performance_parameter import (
    PerformanceParameter,
)
from .pipeline_priority import (
    PipelinePriority,
)
from .prediction_features import (
    PredictionFeatures,
)
from .recovery_strategy import (
    RecoveryStrategy,
)
from .required_capability import (
    RequiredCapability,
)
from .resource_usage import (
    ResourceUsage,
)
from .response_format import (
    ResponseFormat,
)
from .summarization_config import (
    SummarizationConfig,
)
from .system_gpu_stats import (
    SystemGPUStats,
)
from .technical_domain import (
    TechnicalDomain,
)
from .thought import (
    Thought,
)
from .todo_item import (
    TodoItem,
)
from .tool_call import (
    ToolCall,
)
from .tool_config import (
    ToolConfig,
)
from .user_config import (
    UserConfig,
)
from .workflow_config import (
    WorkflowConfig,
)
from .workflow_type import (
    WorkflowType,
)