# Auto-generated model exports
# This file was automatically generated to export all models for easy importing

from __future__ import annotations

# Suppress Pydantic warnings about fields shadowing BaseModel attributes
# (e.g. 'schema' field in OpenAI models shadows deprecated BaseModel.schema())
import warnings
warnings.filterwarnings("ignore", message=".*shadows an attribute in parent.*")

# Import all model modules
try:
    from . import analysis_depth
    from . import api_key
    from . import api_key_request
    from . import api_key_response
    from . import auth_config
    from . import capability_profile_mapping
    from . import chat_req
    from . import chat_response
    from . import circuit_breaker_config
    from . import complexity_estimate
    from . import complexity_level
    from . import computational_requirement
    from . import config
    from . import config_utils
    from . import context_window_config
    from . import conversation
    from . import conversation_ctx
    from . import crash_prevention
    from . import database_config
    from . import default_configs
    from . import default_model_profiles
    from . import dev_stats
    from . import document
    from . import document_source
    from . import dynamic_tool
    from . import embedding_response
    from . import event_stream_config
    from . import execution_state
    from . import generate_req
    from . import generate_response
    from . import generation_state
    from . import gpu_config
    from . import image_generation_config
    from . import image_generation_request
    from . import image_generation_response
    from . import image_metadata
    from . import inference_service
    from . import inference_service_config
    from . import intent
    from . import intent_analysis
    from . import internal_config
    from . import lang_graph_node_state
    from . import lang_graph_state
    from . import learned_limits
    from . import lora_weight
    from . import memory
    from . import memory_config
    from . import memory_fragment
    from . import memory_source
    from . import message
    from . import message_content
    from . import message_content_type
    from . import message_role
    from . import message_type
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
    from . import pagination
    from . import parameter_optimization_config
    from . import parameter_tuning_strategy
    from . import performance_parameter
    from . import performance_parameters
    from . import pipeline_execution_context
    from . import pipeline_execution_state
    from . import pipeline_metrics
    from . import pipeline_priority
    from . import pipeline_state
    from . import prediction_features
    from . import preferences_config
    from . import rabbitmq_config
    from . import recovery_strategy
    from . import redis_config
    from . import refinement_config
    from . import requests
    from . import required_capability
    from . import research_plan
    from . import research_question
    from . import research_question_result
    from . import research_subtask
    from . import research_task
    from . import research_task_status
    from . import resource_usage
    from . import response_format
    from . import routing_strategy
    from . import search_result
    from . import search_result_content
    from . import search_topic_synthesis
    from . import server_config
    from . import socket_connection_type
    from . import socket_message
    from . import socket_session
    from . import socket_stage_type
    from . import socket_status_update
    from . import summarization_config
    from . import summary
    from . import summary_style
    from . import summary_type
    from . import system_gpu_stats
    from . import technical_domain
    from . import thought
    from . import todo_item
    from . import tool
    from . import tool_analysis_request
    from . import tool_call
    from . import tool_config
    from . import tool_needs
    from . import user
    from . import user_config
    from . import web_search_config
    from . import web_socket_connection
    from . import workflow_config
    from . import workflow_type
except ImportError as e:
    import sys
    print(f"Warning: Some model modules could not be imported: {e}", file=sys.stderr)

# Define what gets imported with 'from models import *'
__all__ = [
    'analysis_depth',
    'api_key',
    'api_key_request',
    'api_key_response',
    'auth_config',
    'capability_profile_mapping',
    'chat_req',
    'chat_response',
    'circuit_breaker_config',
    'complexity_estimate',
    'complexity_level',
    'computational_requirement',
    'config',
    'config_utils',
    'context_window_config',
    'conversation',
    'conversation_ctx',
    'crash_prevention',
    'database_config',
    'default_configs',
    'default_model_profiles',
    'dev_stats',
    'document',
    'document_source',
    'dynamic_tool',
    'embedding_response',
    'event_stream_config',
    'execution_state',
    'generate_req',
    'generate_response',
    'generation_state',
    'gpu_config',
    'image_generation_config',
    'image_generation_request',
    'image_generation_response',
    'image_metadata',
    'inference_service',
    'inference_service_config',
    'intent',
    'intent_analysis',
    'internal_config',
    'lang_graph_node_state',
    'lang_graph_state',
    'learned_limits',
    'lora_weight',
    'memory',
    'memory_config',
    'memory_fragment',
    'memory_source',
    'message',
    'message_content',
    'message_content_type',
    'message_role',
    'message_type',
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
    'pagination',
    'parameter_optimization_config',
    'parameter_tuning_strategy',
    'performance_parameter',
    'performance_parameters',
    'pipeline_execution_context',
    'pipeline_execution_state',
    'pipeline_metrics',
    'pipeline_priority',
    'pipeline_state',
    'prediction_features',
    'preferences_config',
    'rabbitmq_config',
    'recovery_strategy',
    'redis_config',
    'refinement_config',
    'requests',
    'required_capability',
    'research_plan',
    'research_question',
    'research_question_result',
    'research_subtask',
    'research_task',
    'research_task_status',
    'resource_usage',
    'response_format',
    'routing_strategy',
    'search_result',
    'search_result_content',
    'search_topic_synthesis',
    'server_config',
    'socket_connection_type',
    'socket_message',
    'socket_session',
    'socket_stage_type',
    'socket_status_update',
    'summarization_config',
    'summary',
    'summary_style',
    'summary_type',
    'system_gpu_stats',
    'technical_domain',
    'thought',
    'todo_item',
    'tool',
    'tool_analysis_request',
    'tool_call',
    'tool_config',
    'tool_needs',
    'user',
    'user_config',
    'web_search_config',
    'web_socket_connection',
    'workflow_config',
    'workflow_type',
    'AnalysisDepth',
    'ApiKey',
    'ApiKeyRequest',
    'ApiKeyResponse',
    'AuthConfig',
    'CapabilityProfileMapping',
    'ChatReq',
    'ChatResponse',
    'CircuitBreakerConfig',
    'ComplexityEstimate',
    'ComplexityLevel',
    'ComputationalRequirement',
    'Config',
    'ContextWindowConfig',
    'Optimization',
    'Prioritization',
    'WindowConfig',
    'Conversation',
    'ConversationCtx',
    'CrashPrevention',
    'DatabaseConfig',
    'DevStats',
    'Document',
    'DocumentSource',
    'DynamicTool',
    'EmbeddingResponse',
    'EventStreamConfig',
    'ExecutionState',
    'GenerateReq',
    'GenerateResponse',
    'GenerationState',
    'GPUConfig',
    'ImageGenerationConfig',
    'ImageGenerateRequest',
    'ImageGenerateResponse',
    'ImageMetadata',
    'InferenceService',
    'InferenceServiceConfig',
    'Intent',
    'IntentAnalysis',
    'InternalConfig',
    'LangGraphNodeState',
    'LangGraphState',
    'LearnedLimits',
    'LoraWeight',
    'Memory',
    'MemoryConfig',
    'MemoryFragment',
    'MemorySource',
    'Message',
    'MessageContent',
    'MessageContentType',
    'MessageRole',
    'MessageType',
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
    'PaginationSchema',
    'ParameterOptimizationConfig',
    'ParameterTuningStrategy',
    'PerformanceParameter',
    'PerformanceParameters',
    'PipelineExecutionContext',
    'PipelineExecutionState',
    'PipelineMetrics',
    'PipelinePriority',
    'PipelineState',
    'PredictionFeatures',
    'PreferencesConfig',
    'RabbitmqConfig',
    'RecoveryStrategy',
    'RedisConfig',
    'RefinementConfig',
    'LoraListResponse',
    'LoraWeightRequest',
    'Malloc',
    'ModelRequest',
    'ModelsListResponse',
    'PromptRequest',
    'RequiredCapability',
    'ResearchPlan',
    'ResearchQuestion',
    'ResearchQuestionResult',
    'ResearchSubtask',
    'ResearchTask',
    'ResearchTaskStatus',
    'ResourceUsage',
    'ResponseFormat',
    'AlternativeStrategy',
    'Metadata',
    'RoutingStrategy',
    'SearchResult',
    'SearchResultContent',
    'SearchTopicSynthesis',
    'ServerConfig',
    'SocketConnectionType',
    'SocketMessage',
    'SocketSession',
    'SocketStageType',
    'SocketStatusUpdate',
    'SummarizationConfig',
    'Summary',
    'SummaryStyle',
    'SummaryType',
    'SystemGPUStats',
    'TechnicalDomain',
    'Thought',
    'TodoItem',
    'Tool',
    'ToolAnalysisRequest',
    'ToolCall',
    'ToolConfig',
    'ToolNeeds',
    'User',
    'UserConfig',
    'WebSearchConfig',
    'WebSocketConnection',
    'WorkflowConfig',
    'WorkflowType',
]

# Re-export all model classes for easy importing and IDE autocompletion
from .analysis_depth import (
    AnalysisDepth,
)
from .api_key import (
    ApiKey,
)
from .api_key_request import (
    ApiKeyRequest,
)
from .api_key_response import (
    ApiKeyResponse,
)
from .auth_config import (
    AuthConfig,
)
from .capability_profile_mapping import (
    CapabilityProfileMapping,
)
from .chat_req import (
    ChatReq,
)
from .chat_response import (
    ChatResponse,
)
from .circuit_breaker_config import (
    CircuitBreakerConfig,
)
from .complexity_estimate import (
    ComplexityEstimate,
)
from .complexity_level import (
    ComplexityLevel,
)
from .computational_requirement import (
    ComputationalRequirement,
)
from .config import (
    Config,
)
from .context_window_config import (
    ContextWindowConfig,
    Optimization,
    Prioritization,
    WindowConfig,
)
from .conversation import (
    Conversation,
)
from .conversation_ctx import (
    ConversationCtx,
)
from .crash_prevention import (
    CrashPrevention,
)
from .database_config import (
    DatabaseConfig,
)
from .dev_stats import (
    DevStats,
)
from .document import (
    Document,
)
from .document_source import (
    DocumentSource,
)
from .dynamic_tool import (
    DynamicTool,
)
from .embedding_response import (
    EmbeddingResponse,
)
from .event_stream_config import (
    EventStreamConfig,
)
from .execution_state import (
    ExecutionState,
)
from .generate_req import (
    GenerateReq,
)
from .generate_response import (
    GenerateResponse,
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
from .image_generation_request import (
    ImageGenerateRequest,
)
from .image_generation_response import (
    ImageGenerateResponse,
)
from .image_metadata import (
    ImageMetadata,
)
from .inference_service import (
    InferenceService,
)
from .inference_service_config import (
    InferenceServiceConfig,
)
from .intent import (
    Intent,
)
from .intent_analysis import (
    IntentAnalysis,
)
from .internal_config import (
    InternalConfig,
)
from .lang_graph_node_state import (
    LangGraphNodeState,
)
from .lang_graph_state import (
    LangGraphState,
)
from .learned_limits import (
    LearnedLimits,
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
from .message_type import (
    MessageType,
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
from .pagination import (
    PaginationSchema,
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
from .performance_parameters import (
    PerformanceParameters,
)
from .pipeline_execution_context import (
    PipelineExecutionContext,
)
from .pipeline_execution_state import (
    PipelineExecutionState,
)
from .pipeline_metrics import (
    PipelineMetrics,
)
from .pipeline_priority import (
    PipelinePriority,
)
from .pipeline_state import (
    PipelineState,
)
from .prediction_features import (
    PredictionFeatures,
)
from .preferences_config import (
    PreferencesConfig,
)
from .rabbitmq_config import (
    RabbitmqConfig,
)
from .recovery_strategy import (
    RecoveryStrategy,
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
from .required_capability import (
    RequiredCapability,
)
from .research_plan import (
    ResearchPlan,
)
from .research_question import (
    ResearchQuestion,
)
from .research_question_result import (
    ResearchQuestionResult,
)
from .research_subtask import (
    ResearchSubtask,
)
from .research_task import (
    ResearchTask,
)
from .research_task_status import (
    ResearchTaskStatus,
)
from .resource_usage import (
    ResourceUsage,
)
from .response_format import (
    ResponseFormat,
)
from .routing_strategy import (
    AlternativeStrategy,
    Metadata,
    RoutingStrategy,
)
from .search_result import (
    SearchResult,
)
from .search_result_content import (
    SearchResultContent,
)
from .search_topic_synthesis import (
    SearchTopicSynthesis,
)
from .server_config import (
    ServerConfig,
)
from .socket_connection_type import (
    SocketConnectionType,
)
from .socket_message import (
    SocketMessage,
)
from .socket_session import (
    SocketSession,
)
from .socket_stage_type import (
    SocketStageType,
)
from .socket_status_update import (
    SocketStatusUpdate,
)
from .summarization_config import (
    SummarizationConfig,
)
from .summary import (
    Summary,
)
from .summary_style import (
    SummaryStyle,
)
from .summary_type import (
    SummaryType,
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
from .tool import (
    Tool,
)
from .tool_analysis_request import (
    ToolAnalysisRequest,
)
from .tool_call import (
    ToolCall,
)
from .tool_config import (
    ToolConfig,
)
from .tool_needs import (
    ToolNeeds,
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
from .web_socket_connection import (
    WebSocketConnection,
)
from .workflow_config import (
    WorkflowConfig,
)
from .workflow_type import (
    WorkflowType,
)