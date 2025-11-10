"""
Production-ready pipeline factory with weakref caching, background cleanup, and
modern/legacy pipeline selection. Replaces the previous garbled version.
"""

import logging
import threading
from typing import Dict, Optional, Type, Union
from contextlib import contextmanager

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel
from models import (
    Model,
    ModelProfile,
    ModelProvider,
    ModelTask,
    PipelinePriority,
)
from .pipeline_cache import LocalPipelineCacheManager
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
        self.logger = logging.getLogger(__name__)

        # Initialize attributes that were removed but are still used
        self._available_models: Dict[str, Model] = ModelLoader().get_available_models()
        self.prefer_langgraph = False  # Default value for langgraph preference
        self._active_loads = 0  # Track active loading operations
        self._active_local_uses = 0  # Track active local pipeline uses

        # Use our new local pipeline cache
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
    ) -> Union[BaseChatModel, Embeddings]:
        model_id = profile.model_name
        model = self._get_model_by_id(model_id)
        if not model:
            raise RuntimeError(f"Model with ID '{model_id}' not found.")

        # DEBUG: Add provider detection logging
        provider = getattr(model, "provider", None)
        import traceback

        call_stack = traceback.extract_stack()[-3:-1]
        call_info = " → ".join(
            [f"{frame.filename.split('/')[-1]}:{frame.lineno}" for frame in call_stack]
        )

        self.logger.info(
            f"🔍 get_pipeline() called for {model_id}, provider={provider}, from: {call_info}"
        )

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
                m: Model, p: ModelProfile, g: Optional[Type[BaseModel]] = grammar
            ) -> Optional[Union[BaseChatModel, Embeddings]]:
                return self.create_pipeline(m, p, g)

            pipeline = self.local_cache.get_or_create(
                model, profile, priority, create_with_coordination, grammar
            )
            if not pipeline:
                raise RuntimeError(
                    f"Failed to create cached pipeline for model '{model.name}'"
                )

            # Automatically lock local pipelines for safety
            locked = self.local_cache.lock_pipeline(model_id)
            if locked:
                self.logger.debug(
                    f"Automatically locked pipeline {model_id} for safe usage"
                )
            else:
                self.logger.warning(
                    f"Could not lock pipeline {model_id} - proceeding without lock"
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
        Unlock a pipeline that was obtained with get_pipeline_safely().

        Only needed if using get_pipeline_safely() instead of the context manager.
        """
        model_id = profile.model_name
        model = self._get_model_by_id(model_id)
        if not model:
            return False

        if self.local_cache.is_local(model):
            return self.local_cache.unlock_pipeline(model_id)

        return True  # Remote pipelines don't need unlocking

    def get_embedding_pipeline(
        self,
        profile: ModelProfile,
        priority: PipelinePriority = PipelinePriority.NORMAL,
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
                m: Model, p: ModelProfile, _g: Optional[Type[BaseModel]] = None
            ) -> Optional[Embeddings]:
                # _g unused: embeddings creation does not use grammar
                return self._create_embedding_pipeline(m, p)

            pipeline = self.local_cache.get_or_create(
                model, profile, priority, create_embedding_fn, None
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

        # Get the pipeline (automatically locked if local provider)
        pipeline = self.get_pipeline(profile, priority, grammar)

        # Check if this is a local provider that was locked
        is_local = self.local_cache.is_local(model)

        # Track usage for coordination
        if is_local:
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
    ) -> Optional[Union[BaseChatModel, Embeddings]]:
        """
        Create a pipeline instance based on model task and pipeline type.
        Args:
            model: Model configuration
            profile: ModelProfile with runtime settings
        Returns:
            An instance of BaseChatModel or Embeddings
        """
        try:
            # DEBUG: Add detailed pipeline creation logging
            import traceback

            call_stack = traceback.extract_stack()[-4:-1]
            call_info = " → ".join(
                [
                    f"{frame.filename.split('/')[-1]}:{frame.lineno}"
                    for frame in call_stack
                ]
            )

            self.logger.info(
                f"🚀 create_pipeline() called for model={model.name}, task={model.task}, pipeline={getattr(model, 'pipeline', 'unknown')}, from: {call_info}"
            )

            if model.task == ModelTask.TEXTTOTEXT:
                self.logger.info(
                    f"🎯 Routing to _create_text_pipeline for {model.name}"
                )
                return self._create_text_pipeline(model, profile, grammar)
            if model.task == ModelTask.VISIONTEXTTOTEXT:
                self.logger.info(
                    f"🎯 Routing to _create_text_pipeline for vision model {model.name}"
                )
                return self._create_text_pipeline(model, profile, grammar)
            if model.task == ModelTask.TEXTTOEMBEDDINGS:
                self.logger.info(
                    f"🎯 Routing to _create_embedding_pipeline for {model.name}"
                )
                return self._create_embedding_pipeline(model, profile)
            if model.task == ModelTask.TEXTTOIMAGE:
                self.logger.info(
                    f"🎯 Routing to _create_image_pipeline for {model.name}"
                )
                return self._create_image_pipeline(model, profile)
            if model.task == ModelTask.IMAGETOIMAGE:
                self.logger.info(
                    f"🎯 Routing to _create_image_to_image_pipeline for {model.name}"
                )
                return self._create_image_to_image_pipeline(model, profile)
            self.logger.error(f"Unsupported task type: {model.task}")
            raise RuntimeError(f"Unsupported task type: {model.task}")
        except Exception as e:
            self.logger.error(f"Error creating pipeline for {model.name}: {e}")

            # Log specific error types for better debugging
            if "unknown model architecture" in str(e):
                self.logger.error(
                    f"Model {model.name} uses unsupported architecture - consider updating llama.cpp or using a different model"
                )
            elif "Failed to create llama_context" in str(e):
                self.logger.error(
                    f"Model {model.name} failed to load - may be corrupted or incompatible"
                )
            elif "validation error" in str(e).lower():
                self.logger.error(f"Model {model.name} configuration validation failed")

            raise

    def _create_text_pipeline(
        self,
        model: Model,
        profile: ModelProfile,
        grammar: Optional[Type[BaseModel]] = None,
    ) -> BaseChatModel:
        self.logger.info(
            f"Creating text pipeline for model: {model.name}, pipeline: {model.pipeline}"
        )
        if model.pipeline == "Qwen3Pipe" or model.pipeline == "Qwen3VLPipeline":
            self.logger.info(f"🔧 Creating Qwen3/Qwen3VL pipeline for {model.name}")

            # Try LangChain-based pipeline first for better tool calling support
            try:
                from .pipelines.llamacpp.langchain_chatopenai_pipeline import (  # pylint: disable=import-outside-toplevel
                    LangChainChatOpenAIPipeline,
                )

                self.logger.info(
                    "Attempting to create LangChainChatOpenAIPipeline for Qwen3"
                )
                pipeline = LangChainChatOpenAIPipeline(model, profile, grammar)
                self.logger.info(
                    "Successfully created LangChainChatOpenAIPipeline for Qwen3"
                )
                return pipeline
            except Exception as langchain_error:
                self.logger.warning(
                    f"LangChainChatOpenAIPipeline creation failed: {langchain_error}"
                )

                # Fallback to original pipeline
                from .pipelines.llamacpp.llamacpp_server_pipeline import (  # pylint: disable=import-outside-toplevel
                    LlamaCppServerPipeline,
                )

                self.logger.info("Falling back to LlamaCppServerPipeline for Qwen3")
                try:
                    pipeline = LlamaCppServerPipeline(model, profile, grammar)
                    self.logger.info(
                        "Successfully created fallback LlamaCppServerPipeline for Qwen3"
                    )
                    return pipeline
                except TypeError as e:
                    self.logger.error(
                        f"Both pipeline types failed for {model.name}: {e}"
                    )
                    raise

        if model.pipeline == "LlamaChatSummPipe":
            self.logger.info(f"🔧 Creating LlamaChatSummPipe for {model.name}")
            from .pipelines.llamacpp.llamacpp_server_pipeline import (  # pylint: disable=import-outside-toplevel
                LlamaCppServerPipeline,
            )

            return LlamaCppServerPipeline(model, profile, grammar)

        if model.pipeline == "OpenAiGptOssPipe":
            self.logger.info(f"🔧 Creating OpenAiGptOssPipe for {model.name}")
            from .pipelines.llamacpp.llamacpp_server_pipeline import (  # pylint: disable=import-outside-toplevel
                LlamaCppServerPipeline,
            )

            return LlamaCppServerPipeline(model, profile, grammar)

        raise RuntimeError(f"Unsupported text pipeline type: {model.pipeline}")

    def _create_embedding_pipeline(
        self,
        model: Model,
        profile: ModelProfile,
    ) -> Optional[Embeddings]:
        if model.pipeline == "NomicEmbedTextPipe":
            try:
                from .pipelines.llamacpp.llamacpp_server_embeddings import (  # pylint: disable=import-outside-toplevel
                    LlamaCppServerEmbeddings,
                )

                return LlamaCppServerEmbeddings(model, profile)
            except Exception as e:
                self.logger.error(f"Failed to initialize LlamaCppServerEmbeddings: {e}")
                return None
        if model.pipeline == "Qwen3EmbeddingPipe":
            try:
                from .pipelines.llamacpp.llamacpp_server_embeddings import (  # pylint: disable=import-outside-toplevel
                    LlamaCppServerEmbeddings,
                )

                return LlamaCppServerEmbeddings(model, profile)
            except Exception as e:
                self.logger.error(f"Failed to initialize LlamaCppServerEmbeddings: {e}")
                return None
        return None

    def _create_image_pipeline(
        self,
        model: Model,
        profile: ModelProfile,
    ) -> Optional[BaseChatModel]:
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
    ) -> Optional[BaseChatModel]:
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
