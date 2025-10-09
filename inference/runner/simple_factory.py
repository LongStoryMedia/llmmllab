"""
Simple Pipeline Factory - Pure LLM interface without orchestration.
Provides clean factory pattern for creating pipeline instances.
"""

import logging
from typing import Dict, Type, Optional, List, Any

from models import Model, ModelProfile, ModelProvider
from runner.pipelines.base import (
    SimplePipelineCore,
    SimpleChatPipeline,
    SimpleEmbeddingPipeline,
    SimpleTextPipeline,
)

logger = logging.getLogger(__name__)


class SimplePipelineFactory:
    """
    Simple factory for creating pipeline instances without orchestration.

    Responsibilities:
    - Create appropriate pipeline instances
    - Validate model/profile combinations
    - Provide clean interface for pipeline access

    Does NOT handle:
    - Graph construction
    - Workflow orchestration
    - Complex routing logic
    - State management
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._registry: Dict[str, Type[SimplePipelineCore]] = {}
        self._instances: Dict[str, SimplePipelineCore] = {}

    def register_pipeline(
        self, pipeline_type: str, pipeline_class: Type[SimplePipelineCore]
    ) -> None:
        """Register a pipeline type with the factory."""
        self._registry[pipeline_type] = pipeline_class
        self.logger.info(f"Registered pipeline type: {pipeline_type}")

    def create_pipeline(
        self, model: Model, profile: ModelProfile, pipeline_type: Optional[str] = None
    ) -> SimplePipelineCore:
        """
        Create a pipeline instance for the given model and profile.

        Args:
            model: Model configuration
            profile: Model profile configuration
            pipeline_type: Optional specific pipeline type to create

        Returns:
            SimplePipelineCore: Configured pipeline instance
        """
        try:
            # Generate cache key
            cache_key = f"{model.id}_{profile.name}_{pipeline_type or 'auto'}"

            # Return cached instance if available
            if cache_key in self._instances:
                self.logger.debug(f"Using cached pipeline: {cache_key}")
                return self._instances[cache_key]

            # Determine pipeline type if not specified
            if pipeline_type is None:
                pipeline_type = self._infer_pipeline_type(model, profile)

            # Get pipeline class
            if pipeline_type not in self._registry:
                raise ValueError(f"Unknown pipeline type: {pipeline_type}")

            pipeline_class = self._registry[pipeline_type]

            # Create and cache instance
            pipeline = pipeline_class(model, profile)
            self._instances[cache_key] = pipeline

            self.logger.info(
                f"Created pipeline: {pipeline_type} for model {model.name}"
            )

            return pipeline

        except Exception as e:
            self.logger.error(f"Failed to create pipeline: {e}")
            raise

    def _infer_pipeline_type(self, model: Model, profile: ModelProfile) -> str:
        """
        Infer the appropriate pipeline type based on model and profile.

        Args:
            model: Model configuration
            profile: Model profile configuration (currently unused but available for future logic)

        Returns:
            str: Inferred pipeline type
        """
        # Simple heuristics for pipeline type inference
        model_name = model.name.lower()

        # Could use profile for more sophisticated inference in the future
        _ = profile  # Acknowledge parameter for future use

        # Check for external provider models first
        if hasattr(model, "provider"):
            if model.provider == ModelProvider.OPENAI:
                return "openai"
            elif model.provider == ModelProvider.ANTHROPIC:
                return "anthropic"

        # Check for specific local models
        if "openai_gpt_oss" in model_name or "openai-gpt-oss" in model_name:
            return "openai_gpt_oss"
        elif "qwen" in model_name:
            if "vl" in model_name or "vision" in model_name:
                return "qwen_vision"
            elif "moe" in model_name or "3moe" in model_name:
                return "qwen_moe"
            else:
                return "qwen"
        elif "llama" in model_name and "chat" in model_name and "summ" in model_name:
            return "llama_chat_sum"
        elif "nomic" in model_name and "embed" in model_name:
            return "nomic_embed"

        # Check for embedding models
        if any(
            keyword in model_name for keyword in ["embed", "embedding", "bge", "e5"]
        ):
            return "embedding"

        # Check for text/completion models
        if any(
            keyword in model_name for keyword in ["text", "completion", "summarize"]
        ):
            return "text"

        # Default to chat for most models
        return "chat"

    def get_available_pipelines(self) -> List[str]:
        """Get list of available pipeline types."""
        return list(self._registry.keys())

    def cleanup_all(self) -> None:
        """Clean up all cached pipeline instances."""
        for pipeline in self._instances.values():
            try:
                pipeline.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up pipeline: {e}")

        self._instances.clear()
        self.logger.info("All pipelines cleaned up")

    def get_pipeline_info(self, pipeline_type: str) -> Dict[str, Any]:
        """Get information about a specific pipeline type."""
        if pipeline_type not in self._registry:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")

        pipeline_class = self._registry[pipeline_type]

        return {
            "type": pipeline_type,
            "class": pipeline_class.__name__,
            "module": pipeline_class.__module__,
            "allowed_return_types": [
                t.__name__ for t in pipeline_class.allowed_return_types
            ],
            "default_return_type": (
                pipeline_class.default_return_type.__name__
                if pipeline_class.default_return_type
                else None
            ),
        }


# Create default factory instance
default_factory = SimplePipelineFactory()

# Register built-in pipeline types
default_factory.register_pipeline("chat", SimpleChatPipeline)
default_factory.register_pipeline("embedding", SimpleEmbeddingPipeline)
default_factory.register_pipeline("text", SimpleTextPipeline)

# Register specific model implementations
try:
    from runner.pipelines.txt2txt.openai_gpt_oss import OpenAIGptOssSimplePipeline

    default_factory.register_pipeline("openai_gpt_oss", OpenAIGptOssSimplePipeline)
except ImportError as e:
    logger.warning(f"Could not register openai_gpt_oss pipeline: {e}")

try:
    from runner.pipelines.txt2txt.qwen3moe import Qwen3Moe

    default_factory.register_pipeline("qwen_moe", Qwen3Moe)
except ImportError as e:
    logger.warning(f"Could not register qwen_moe pipeline: {e}")

try:
    from runner.pipelines.txt2txt.llamachatsum import LlamaChatSummPipe

    default_factory.register_pipeline("llama_chat_sum", LlamaChatSummPipe)
except ImportError as e:
    logger.warning(f"Could not register llama_chat_sum pipeline: {e}")

try:
    from runner.pipelines.emb.nom2 import NomicEmbedTextPipe

    default_factory.register_pipeline("nomic_embed", NomicEmbedTextPipe)
except ImportError as e:
    logger.warning(f"Could not register nomic_embed pipeline: {e}")

# Register external provider pipelines
try:
    from runner.pipelines.external.openai_pipeline import OpenAIPipeline

    default_factory.register_pipeline("openai", OpenAIPipeline)
except ImportError as e:
    logger.warning(f"Could not register openai pipeline: {e}")

try:
    from runner.pipelines.external.anthropic_pipeline import AnthropicPipeline

    default_factory.register_pipeline("anthropic", AnthropicPipeline)
except ImportError as e:
    logger.warning(f"Could not register anthropic pipeline: {e}")
