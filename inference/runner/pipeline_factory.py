"""
Production-ready pipeline factory with weakref caching, background cleanup, and
modern/legacy pipeline selection. Replaces the previous garbled version.
"""

import json
import yaml
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Type, Union
from contextlib import contextmanager

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel
from models import (
    Model,
    LoraWeight,
    ModelDetails,
    ModelProfile,
    ModelProvider,
    ModelTask,
    PipelinePriority,
)
from .pipeline_cache import LocalPipelineCacheManager


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
        self._available_models: Dict[str, Model] = {}
        self.prefer_langgraph = False  # Default value for langgraph preference
        self._active_loads = 0  # Track active loading operations
        self._active_local_uses = 0  # Track active local pipeline uses

        # Use our new local pipeline cache
        self.local_cache = LocalPipelineCacheManager()

        # Coordination for memory-constrained loading
        self._coord_lock = threading.Lock()
        self._coord_cond = threading.Condition(self._coord_lock)

        # Track models currently being loaded to prevent duplicate loading
        self._loading_models: Dict[str, threading.Event] = (
            {}
        )  # model_id -> Event to wait for loading completion

        # Load available models from config file
        self._load_available_models()

        # Set self.models to the loaded models, with models_map as fallback
        self.models: Dict[str, Model] = (
            self._available_models if self._available_models else (models_map or {})
        )

        self.logger.info("PipelineFactory initialized with LocalPipelineCacheManager")

    # Background cleanup is handled entirely in LocalPipelineCacheManager

    # ---------- Model loading ----------

    def _load_available_models(self) -> None:
        """Load available models from YAML (preferred) or JSON (legacy) configuration.

        Order of precedence:
        1. MODELS_FILE_PATH env var (can point to YAML or JSON)
        2. /app/.models.yaml
        3. /app/.models.json (legacy fallback)
        """
        candidates: List[str] = []
        env_path = os.environ.get("MODELS_FILE_PATH")
        if env_path:
            candidates.append(env_path)
        candidates.extend(["/app/.models.yaml", "/app/.models.json"])

        chosen_path: Optional[str] = None
        for path in candidates:
            if path and os.path.exists(path):
                chosen_path = path
                break

        if not chosen_path:
            self.logger.error(
                "No models configuration file found (checked env, .models.yaml, .models.json)"
            )
            return

        # Attempt parsing based on extension; allow YAML for .yaml/.yml, otherwise JSON
        try:
            with open(chosen_path, "r", encoding="utf-8") as f:
                if chosen_path.endswith((".yaml", ".yml")):
                    models_data = yaml.safe_load(f)
                else:
                    models_data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to parse models file {chosen_path}: {e}")
            return

        if not isinstance(models_data, list):
            self.logger.error(
                f"Models config {chosen_path} is not a list; ignoring contents"
            )
            return

        loaded_count = 0
        for data in models_data:
            try:
                model = self._create_model_from_data(data)
                if model:
                    id_key = str(data.get("id") or model.id or "")
                    if not id_key:
                        self.logger.error(
                            f"Skipping model with missing id: {getattr(model, 'name', 'unknown')}"
                        )
                        continue
                    self._available_models[id_key] = model
                    loaded_count += 1
            except Exception as e:
                self.logger.error(
                    f"Error creating model from {data.get('id', 'unknown')}: {e}"
                )

        self.logger.info(
            f"Loaded {loaded_count}/{len(models_data)} models from config ({os.path.basename(chosen_path)})"
        )

    def _create_model_from_data(self, data: Dict[str, Any]) -> Optional[Model]:
        # LoRA weights
        loras: List[LoraWeight] = []
        for lw in data.get("lora_weights", []) or []:
            try:
                loras.append(
                    LoraWeight(
                        id=lw.get("id", ""),
                        name=lw.get("name", ""),
                        weight_name=lw.get("weight_name", ""),
                        adapter_name=lw.get("adapter_name", ""),
                        parent_model=lw.get("parent_model", ""),
                    )
                )
            except Exception:
                continue

        details_dict = data.get("details", {}) or {}
        try:
            # Map specialization values to valid schema enum values
            specialization = details_dict.get("specialization")
            if specialization:
                # Map invalid specialization values to valid ones
                specialization_map = {
                    "ImageTextToText": "Text",
                    "TextGeneration": "Text",
                    "TextSummarization": "Text",
                    "TextToText": "Text",
                    "TextToEmbeddings": "Embedding",
                    # Valid values remain unchanged
                    "Text": "Text",
                    "LoRA": "LoRA",
                    "Embedding": "Embedding",
                    "TextToImage": "TextToImage",
                    "ImageToImage": "ImageToImage",
                    "Audio": "Audio",
                }
                specialization = specialization_map.get(
                    specialization, "Text"
                )  # Default to "Text"

            details = ModelDetails(
                parent_model=details_dict.get("parent_model"),
                format=str(details_dict.get("format", "")),
                family=str(details_dict.get("family", "")),
                families=list(details_dict.get("families", [])),
                parameter_size=str(details_dict.get("parameter_size", "")),
                quantization_level=details_dict.get("quantization_level"),
                specialization=specialization,  # type: ignore[arg-type]
                dtype=str(details_dict.get("dtype", "bf16")),
                precision=str(details_dict.get("precision", "fp16")),  # type: ignore[arg-type]
                weight=float(details_dict.get("weight", 1.0)),
                gguf_file=details_dict.get("gguf_file"),
                clip_model_path=details_dict.get("clip_model_path"),
                description=details_dict.get("description"),
            )
        except Exception as e:
            self.logger.error(f"Invalid model details for {data.get('id')}: {e}")
            return None

        try:
            model = Model(
                id=data.get("id"),
                name=data["name"],
                model=data["model"],
                provider=data["provider"],
                modified_at=data["modified_at"],
                size=data["size"],
                digest=data["digest"],
                pipeline=data.get("pipeline"),
                lora_weights=loras,
                details=details,
                task=data.get("task", "TextToText"),
            )
        except Exception as e:
            self.logger.error(f"Invalid model entry: {e}")
            return None

        return model

    # ---------- Public API ----------

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

        # Local providers -> managed cached path with automatic locking
        if getattr(model, "provider", None) in {
            ModelProvider.LLAMA_CPP,
            ModelProvider.STABLE_DIFFUSION_CPP,
        }:
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
        pipeline = self.create_pipeline(model, profile)
        if not pipeline:
            raise RuntimeError(
                f"Failed to create pipeline for model '{model.name}' (provider: {getattr(model, 'provider', 'unknown')})"
            )
        self.logger.debug(
            f"Created transient pipeline for remote provider {getattr(model, 'provider', 'unknown')} ({model.name})"
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
            self.logger.info(f"Creating pipeline for {model.name} (task: {model.task})")

            if model.task == ModelTask.TEXTTOTEXT:
                return self._create_text_pipeline(model, profile, grammar)
            if model.task == ModelTask.VISIONTEXTTOTEXT:
                return self._create_text_pipeline(model, profile, grammar)
            if model.task == ModelTask.TEXTTOEMBEDDINGS:
                return self._create_embedding_pipeline(model, profile)
            if model.task == ModelTask.TEXTTOIMAGE:
                return self._create_image_pipeline(model, profile)
            if model.task == ModelTask.IMAGETOIMAGE:
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
        if model.pipeline == "Qwen3Pipe":
            self.logger.info(
                f"Creating Qwen pipeline, prefer_langgraph={self.prefer_langgraph}"
            )
            from .pipelines.txt2txt.qwen3moe import (  # pylint: disable=import-outside-toplevel
                Qwen3Moe,
            )

            self.logger.info("Attempting to create Qwen3Moe")
            try:
                # Try with expected_return_type first (preferred)
                pipeline = Qwen3Moe(model, profile, grammar)
            except TypeError as e:
                self.logger.warning(f"Qwen3Moe creation failed: {e}")
                raise
            self.logger.info("Successfully created Qwen3Moe")
            return pipeline

        if model.pipeline == "Qwen25VLGGUFPipeline":
            # File may not exist; fallback handled below
            from .pipelines.imgtxt2txt.qwen25_vl import (  # pylint: disable=import-outside-toplevel
                Qwen25VLPipeline,
            )

            return Qwen25VLPipeline(model, profile, grammar)

        if model.pipeline == "Qwen3VLPipeline":
            from .pipelines.imgtxt2txt.qwen3_vl import (  # pylint: disable=import-outside-toplevel
                Qwen3VLPipeline,
            )

            return Qwen3VLPipeline(model, profile, grammar)

        if model.pipeline == "LlamaChatSummPipe":
            from .pipelines.txt2txt.llamachatsum import (  # pylint: disable=import-outside-toplevel
                LlamaChatSummPipe,
            )

            return LlamaChatSummPipe(model, profile, grammar)

        if model.pipeline == "OpenAiGptOssPipe":
            from .pipelines.txt2txt.openai_gpt_oss import (  # pylint: disable=import-outside-toplevel
                OpenAIGptOssPipeline,
            )

            return OpenAIGptOssPipeline(model, profile, grammar)

        raise RuntimeError(f"Unsupported text pipeline type: {model.pipeline}")

    def _create_embedding_pipeline(
        self,
        model: Model,
        profile: ModelProfile,
    ) -> Optional[Embeddings]:
        if model.pipeline == "NomicEmbedTextPipe":
            try:
                from .pipelines.emb.nomic_embeddings import (  # pylint: disable=import-outside-toplevel
                    NomicEmbeddings,
                )

                return NomicEmbeddings(model, profile)
            except Exception as e:
                self.logger.error(f"Failed to initialize NomicEmbeddings: {e}")
                return None
        if model.pipeline == "Qwen3EmbeddingPipe":
            try:
                from .pipelines.emb.qwen3_embeddings import (  # pylint: disable=import-outside-toplevel
                    Qwen3Embeddings,
                )

                return Qwen3Embeddings(model, profile)
            except Exception as e:
                self.logger.error(f"Failed to initialize Qwen3Embeddings: {e}")
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
