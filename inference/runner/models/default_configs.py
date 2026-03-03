"""
Default configuration objects for user settings.

This file contains predefined default objects for each configuration type.
When schemas are updated and models regenerated, the linter will indicate
where these defaults need to be updated.
"""

import uuid

from runner.models.preferences_config import PreferencesConfig
from runner.models.memory_config import MemoryConfig
from runner.models.summarization_config import SummarizationConfig
from runner.models.refinement_config import RefinementConfig
from runner.models.web_search_config import WebSearchConfig
from runner.models.image_generation_config import ImageGenerationConfig
from runner.models.circuit_breaker_config import CircuitBreakerConfig
from runner.models.gpu_config import GPUConfig
from runner.models.user_config import UserConfig
from runner.models.workflow_config import WorkflowConfig
from runner.models.tool_config import ToolConfig
from runner.models.parameter_optimization_config import (
    ParameterOptimizationConfig,
    PerformanceParameter,
    ParameterTuningStrategy,
)
from runner.models.crash_prevention import CrashPrevention
from runner.models.context_window_config import (
    ContextWindowConfig,
    WindowConfig,
    Prioritization,
    Optimization,
)
from runner.models.model_profile_config import ModelProfileConfig

# Removed circular import - DEFAULT_MODEL_PROFILE_CONFIG created inline below