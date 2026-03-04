# Composer models - all models used by composer are defined here
# This file re-exports all models for convenient importing

from __future__ import annotations

# Suppress Pydantic warnings about fields shadowing BaseModel attributes
import warnings

warnings.filterwarnings("ignore", message=".*shadows an attribute in parent.*")

# Import all model modules
from . import chat_response
from . import circuit_breaker_config
from . import complexity_level
from . import computational_requirement
from . import crash_prevention
from . import context_window_config
from . import document
from . import dynamic_tool
from . import event_stream_config
from . import generation_state
from . import gpu_config
from . import image_generation_config
from . import intent_analysis
from . import memory
from . import memory_config
from . import memory_fragment
from . import memory_source
from . import message
from . import message_content_type
from . import message_role
from . import model_parameters
from . import model_profile
from . import model_profile_config
from . import model_profile_type
from . import node_metadata
from . import parameter_optimization_config
from . import parameter_tuning_strategy
from . import performance_parameter
from . import preferences_config
from . import refinement_config
from . import required_capability
from . import resource_usage
from . import response_format
from . import search_result
from . import search_result_content
from . import search_topic_synthesis
from . import summarization_config
from . import summary
from . import technical_domain
from . import todo_item
from . import tool
from . import tool_call
from . import tool_config
from . import user_config
from . import web_search_config
from . import workflow_config
from . import workflow_type

# Re-export all model classes for easy importing and IDE autocompletion
from .chat_response import ChatResponse
from .circuit_breaker_config import CircuitBreakerConfig
from .complexity_level import ComplexityLevel
from .computational_requirement import ComputationalRequirement
from .crash_prevention import CrashPrevention
from .context_window_config import ContextWindowConfig
from .document import Document
from .event_stream_config import EventStreamConfig
from .generation_state import GenerationState
from .gpu_config import GPUConfig
from .image_generation_config import ImageGenerationConfig
from .intent_analysis import IntentAnalysis
from .memory import Memory
from .memory_config import MemoryConfig
from .memory_fragment import MemoryFragment
from .memory_source import MemorySource
from .message import Message
from .message_content_type import MessageContentType
from .message_role import MessageRole
from .model_parameters import ModelParameters
from .model_profile import ModelProfile
from .model_profile_config import ModelProfileConfig
from .model_profile_type import ModelProfileType
from .node_metadata import NodeMetadata
from .parameter_optimization_config import ParameterOptimizationConfig
from .parameter_tuning_strategy import ParameterTuningStrategy
from .performance_parameter import PerformanceParameter
from .preferences_config import PreferencesConfig
from .refinement_config import RefinementConfig
from .required_capability import RequiredCapability
from .resource_usage import ResourceUsage
from .response_format import ResponseFormat
from .search_result import SearchResult
from .search_result_content import SearchResultContent
from .search_topic_synthesis import SearchTopicSynthesis
from .summarization_config import SummarizationConfig
from .summary import Summary
from .technical_domain import TechnicalDomain
from .todo_item import TodoItem
from .tool import Tool
from .tool_config import ToolConfig
from .user_config import UserConfig
from .web_search_config import WebSearchConfig
from .workflow_config import WorkflowConfig
from .workflow_type import WorkflowType

# Re-export additional models used by composer
from .dynamic_tool import DynamicTool
from .message_content import MessageContent
from .thought import Thought
from .tool_call import ToolCall

# Default configurations
from .default_configs import DEFAULT_MEMORY_CONFIG, DEFAULT_WEB_SEARCH_CONFIG
