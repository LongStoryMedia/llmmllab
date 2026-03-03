"""
Production-ready pipeline factory with weakref caching, background cleanup, and
modern/legacy pipeline selection. Replaces the previous garbled version.
"""

import threading
from typing import Dict, Optional, Type, Union, Any
from contextlib import contextmanager

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel
from runner.models import (
    Model,
    ModelProfile,
    ModelProvider,
    ModelTask,
    PipelinePriority,
)
from runner.pipelines.base import BasePipeline
from runner.utils.logging import llmmllogger
from .pipeline_cache import LocalPipelineCacheManager

try:
    # Prefer the shared module-global cache if available
    from .pipeline_cache import local_pipeline_cache as _GLOBAL_PIPELINE_CACHE
except Exception:
    _GLOBAL_PIPELINE_CACHE = None
from .utils.model_loader import ModelLoader


class PipelineFactory:
    """
    Factory for creating pipelines.

    Handles:
    - Pipeline creation and coordination
    - Resource allocation coordination
    - Delegating cache management to LocalPipelineCacheManager
    """

    def __init__(self, models_map: Dict[str, Model]):
        self.logger = llmmllogger.bind(component="PipelineFactory")

        # Initialize attributes that were removed but are still used
        self._available_models: Dict[str, Model] = ModelLoader().get_available_models()
        self.prefer_langgraph = False  # Default value for langgraph preference
        self._active_loads = 0  # Track active loading operations
        self._active_local_uses = 0  # Track active local pipeline uses

        # Use the shared module-global cache if present, otherwise create one
        if _GLOBAL_PIPELINE_CACHE is not None:
            self.local_cache = _GLOBAL_PIPELINE_CACHE
        else:
            self.local_cache = LocalPipelineCacheManager()

        # Coordination for memory-constrained loading
        self._coord_lock = threading.Lock()
        self._coord_cond = threading.Condition(self._coord_lock)

        # Set self.models to the loaded models, with models_map as fallback
        self.models: Dict[str, Model] = (
            self._available_models if self._available_models else (models_map or {})
        )

        self.logger.info("PipelineFactory initialized with LocalPipelineCacheManager")

    def get_pipeline(
        self,
        profile: ModelProfile,
        priority: PipelinePriority = PipelinePriority.NORMAL,
        grammar: Optional[Type[BaseModel]] = None,
        metadata: Optional[dict] = {},
    ) -> Union[BasePipeline, Embeddings]:
        model_id = profile.model_name
        self.logger.debug(
            f"Requesting pipeline for model_id: {model_id}, priority: {priority}, grammar: {grammar}, metadata: {metadata}"
        )
        model = self._get_model_by_id(model_id)
        if not model:
            raise RuntimeError(f"Model with ID '{model_id}' not found.")

        # DEBUG: Add provider detection logging
        provider = getattr(model, "provider", None)
        # Local providers -> managed cached path with automatic locking
        if provider in {
            ModelProvider.LLAMA_CPP,
            ModelProvider.STABLE_DIFFUSION_CPP,
        }:
            self.logger.info(
                f"📦 Using LOCAL cached path for {model_id} (provider: {provider})"
            )

            # Use a factory function that handles coordination internally
            def create_with_coordination(
                m: Model,
                p: ModelProfile,
                g: Optional[Type[BaseModel]] = grammar,
                metadata: Optional[dict] = {},
            ) -> Optional[Union[BasePipeline, Embeddings]]:
                return self.create_pipeline(m, p, g, metadata)

            pipeline = self.local_cache.get_or_create(
                model,
                profile,
                priority,
                create_with_coordination,
                grammar,
                metadata=metadata,
            )
            if not pipeline:
                raise RuntimeError(
                    f"Failed to create cached pipeline for model '{model.name}'"
                )

            return pipeline

        # Remote / API providers -> create transient each call, no caching or locking needed
        self.logger.info(
            f"🌐 Using REMOTE non-cached path for {model_id} (provider: {provider})"
        )
        pipeline = self.create_pipeline(model, profile)
        if not pipeline:
            raise RuntimeError(
                f"Failed to create pipeline for model '{model.name}' (provider: {provider})"
            )
        self.logger.debug(
            f"Created transient pipeline for remote provider {provider} ({model.name})"
        )
        return pipeline

    def unlock_pipeline(self, profile: ModelProfile) -> bool:
        """
        Unlock a pipeline that was obtained with get_pipeline().

        The pipeline remains cached and available for reuse by other components.
        Only removes the exclusive lock, does not evict from cache.
        """
        model_id = profile.model_name
        model = self._get_model_by_id(model_id)
        if not model:
            return False

        if self.local_cache.is_local(model):
            return self.local_cache.unlock_pipeline(model_id)

        return True  # Remote pipelines don't need unlocking

    def set_pipeline_persistent(
        self, profile: ModelProfile, persistent: bool = True
    ) -> bool:
        """Mark a pipeline as persistent to prevent eviction unless absolutely necessary."""
        model_id = profile.model_name
        model = self._get_model_by_id(model_id)
        if not model:
            return False

        if self.local_cache.is_local(model):
            return self.local_cache.set_persistent(model_id, persistent)

        return True  # Remote pipelines are always transient

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics for monitoring."""
        return self.local_cache.get_cache_info()

    def force_evict_pipeline(self, profile: ModelProfile) -> bool:
        """Force eviction of a specific pipeline from cache."""
        model_id = profile.model_name
        model = self._get_model_by_id(model_id)
        if not model:
            return False

        if self.local_cache.is_local(model):
            self.local_cache.clear_cache(model_id)
            return True

        return False

    def get_embedding_pipeline(
        self,
        profile: ModelProfile,
        priority: PipelinePriority = PipelinePriority.NORMAL,
        metadata: Optional[dict] = None,
    ) -> Embeddings:
        """Get specifically an embedding pipeline with proper typing."""
        model_id = profile.model_name
        model = self._get_model_by_id(model_id)
        if not model:
            raise RuntimeError(f"Model with ID '{model_id}' not found.")

        # For embedding models, require embedding-specific task
        if model.task != "TextToEmbeddings":
            raise ValueError(
                f"Model '{model.name}' is not an embedding model (task: {model.task})"
            )

        # Local providers -> managed cached path
        if getattr(model, "provider", None) in {
            ModelProvider.LLAMA_CPP,
            ModelProvider.STABLE_DIFFUSION_CPP,
        }:

            def create_embedding_fn(
                m: Model,
                p: ModelProfile,
                _g: Optional[Type[BaseModel]] = None,
                metadata: Optional[dict] = None,
            ) -> Optional[Embeddings]:
                # _g unused: embeddings creation does not use grammar
                return self._create_embedding_pipeline(m, p, metadata=metadata)

            pipeline = self.local_cache.get_or_create(
                model, profile, priority, create_embedding_fn, None, metadata=metadata
            )
            if not pipeline:
                raise RuntimeError(
                    f"Failed to create cached embedding pipeline for model '{model.name}'"
                )
            if not isinstance(pipeline, Embeddings):
                raise ValueError(f"Expected Embeddings instance, got {type(pipeline)}")
            return pipeline

        # Remote / API providers -> create transient each call, no caching
        pipeline = self._create_embedding_pipeline(model, profile)
        if not pipeline:
            raise RuntimeError(
                f"Failed to create embedding pipeline for model '{model.name}' (provider: {getattr(model, 'provider', 'unknown')})"
            )
        return pipeline

    @contextmanager
    def pipeline(
        self,
        profile: ModelProfile,
        priority: PipelinePriority = PipelinePriority.NORMAL,
        grammar: Optional[Type[BaseModel]] = None,
    ):
        """
        Context manager for safe pipeline usage with automatic locking and unlocking.

        get_pipeline() automatically locks local providers, this context manager
        ensures proper unlocking when done.
        """
        model_id = profile.model_name
        model = self._get_model_by_id(model_id)
        if not model:
            raise RuntimeError(f"Model with ID '{model_id}' not found.")

        pipeline = self.get_pipeline(profile, priority, grammar)
        is_local = self.local_cache.is_local(model)

        # Explicitly lock local pipelines to prevent eviction during use
        if is_local:
            self.local_cache.lock_pipeline(model_id)
            with self._coord_cond:
                self._active_local_uses += 1

        try:
            yield pipeline
        finally:
            if is_local:
                # Unlock the pipeline and update coordination
                self.local_cache.unlock_pipeline(model_id)
                with self._coord_cond:
                    self._active_local_uses = max(0, self._active_local_uses - 1)
                    self._coord_cond.notify_all()

    def _get_model_by_id(self, model_id: str) -> Optional[Model]:
        if not self._available_models:
            self.logger.error("Available models dictionary is empty")
            return None
        if model_id not in self._available_models:
            self.logger.error(
                f"Model '{model_id}' not found. Available: {list(self._available_models.keys())}"
            )
            return None
        return self._available_models[model_id]

    def create_pipeline(
        self,
        model: Model,
        profile: ModelProfile,
        grammar: Optional[Type[BaseModel]] = None,
        metadata: Optional[dict] = {},
    ) -> Optional[Union[BasePipeline, Embeddings]]:
        """
        Create a pipeline instance based on model task and pipeline type.
        Args:
            model: Model configuration
            profile: ModelProfile with runtime settings
        Returns:
            An instance of BaseChatModel or Embeddings
        """
        try:
            if (
                model.task == ModelTask.TEXTTOTEXT
                or model.task == ModelTask.VISIONTEXTTOTEXT
            ):
                self.logger.info(
                    f"🎯 Routing to _create_text_pipeline for vision model {model.name}"
                )
                return self._create_text_pipeline(model, profile, grammar, metadata)
            if model.task == ModelTask.TEXTTOEMBEDDINGS:
                self.logger.info(
                    f"🎯 Routing to _create_embedding_pipeline for {model.name}"
                )
                return self._create_embedding_pipeline(model, profile, metadata)
            if model.task == ModelTask.TEXTTOIMAGE:
                self.logger.info(
                    f"🎯 Routing to _create_image_pipeline for {model.name}"
                )
                return self._create_image_pipeline(model, profile, metadata)
            if model.task == ModelTask.IMAGETOIMAGE:
                self.logger.info(
                    f"🎯 Routing to _create_image_to_image_pipeline for {model.name}"
                )
                return self._create_image_to_image_pipeline(model, profile, metadata)
            self.logger.error(f"Unsupported task type: {model.task}")
            raise RuntimeError(f"Unsupported task type: {model.task}")
        except Exception as e:
            self.logger.error(f"Error creating pipeline for {model.name}: {e}")
            raise

    def _create_text_pipeline(
        self,
        model: Model,
        profile: ModelProfile,
        grammar: Optional[Type[BaseModel]] = None,
        metadata: Optional[dict] = {},
    ) -> BasePipeline:
        self.logger.info(
            f"Creating text pipeline for model: {model.name}, pipeline: {model.pipeline}, provider: {model.provider}"
        )

        match model.provider:
            case ModelProvider.LLAMA_CPP:
                from .pipelines.llamacpp.chat import (
                    ChatLlamaCppPipeline,
                )  # pylint: disable=import-outside-toplevel

                return ChatLlamaCppPipeline(model, profile, grammar, metadata)
            case ModelProvider.OPENAI:
                import os
                from langchain_openai import (
                    ChatOpenAI,
                )  # pylint: disable=import-outside-toplevel
                from pydantic import SecretStr

                return ChatOpenAI(  # type: ignore[return-value]
                    model=model.name,
                    api_key=SecretStr(os.environ.get("OPENAI_API_KEY", "")),
                )
            case ModelProvider.ANTHROPIC:
                import os
                from langchain_anthropic import (
                    ChatAnthropic,
                )  # pylint: disable=import-outside-toplevel
                from pydantic import SecretStr

                return ChatAnthropic(  # type: ignore[return-value]
                    model_name=model.name,
                    api_key=SecretStr(os.environ.get("ANTHROPIC_API_KEY", "")),
                )
            case _:
                raise ValueError(f"Unsupported text provider: {model.provider}")

    def _create_embedding_pipeline(
        self,
        model: Model,
        profile: ModelProfile,
        metadata: Optional[dict] = {},
    ) -> Optional[Embeddings]:
        from .pipelines.llamacpp.embed import (  # pylint: disable=import-outside-toplevel
            EmbedLlamaCppPipeline,
        )

        return EmbedLlamaCppPipeline(model, profile, metadata=metadata)

    def _create_image_pipeline(
        self,
        model: Model,
        profile: ModelProfile,
        metadata: Optional[dict] = {},
    ) -> Optional[BasePipeline]:
        if model.pipeline == "FluxPipeline":
            try:
                from .pipelines.txt2img.flux import (  # pylint: disable=import-outside-toplevel
                    FluxPipe,
                )

                return FluxPipe(  # pylint: disable=abstract-class-instantiated
                    model, profile
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize FluxPipe: {e}")
                return None
        return None

    def _create_image_to_image_pipeline(
        self,
        model: Model,
        profile: ModelProfile,
        metadata: Optional[dict] = {},
    ) -> Optional[BasePipeline]:
        if model.pipeline == "FluxKontextPipeline":
            try:
                from .pipelines.img2img.flux import (  # pylint: disable=import-outside-toplevel
                    FluxKontextPipe,
                )

                return FluxKontextPipe(  # pylint: disable=abstract-class-instantiated
                    model, profile
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize FluxKontextPipe: {e}")
                return None
        return None

    # (Removed duplicate legacy cleanup method; single alias earlier in file)


# Create global factory instance
pipeline_factory = PipelineFactory({})
