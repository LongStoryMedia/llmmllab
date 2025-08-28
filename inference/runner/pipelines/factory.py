"""
Updated factory.py to integrate LangGraph-based pipelines.
This maintains the existing factory pattern while supporting both legacy and modern pipeline implementations.
"""

import json
import logging
import os
import time
import threading
from typing import Any, Dict, List, Optional, Type, cast

from models import Model, LoraWeight, ModelDetails, ModelProfile, ChatResponse

# from .base_dual_pipeline import BasePipelineDual, BasePipelineCore
from .base_pipeline import BasePipelineCore


class PipelineCacheEntry:
    """A class to store BasePipeline cache entries with timeout information."""

    def __init__(self, pipeline: BasePipelineCore, timestamp: Optional[float] = None):
        self.pipeline = pipeline
        self.last_accessed = timestamp if timestamp is not None else time.time()


class ModernPipelineFactory:
    """
    Enhanced factory class supporting both legacy AgentExecutor and modern LangGraph pipelines.

    This factory automatically determines the appropriate pipeline implementation based on model
    configuration and maintains backward compatibility while enabling modern LangGraph features.

    Key features:
    - Automatic pipeline type detection and selection
    - Backward compatibility with existing AgentExecutor pipelines
    - Modern LangGraph integration for improved performance and control
    - Configurable timeout and cleanup mechanisms
    - Resource management for GPU memory optimization
    """

    # Cache for loaded pipelines
    _pipelines: Dict[str, PipelineCacheEntry] = {}
    _available_models: Dict[str, Model] = {}

    # Cache configuration
    _cache_timeout = 300  # 5 minutes default
    _cleanup_thread = None
    _cleanup_lock = threading.RLock()

    def __init__(self, prefer_langgraph: bool = True):
        """
        Initialize the factory.

        Args:
            prefer_langgraph: If True, use LangGraph implementations when available.
                             If False, prefer legacy AgentExecutor implementations.
        """
        self.logger = logging.getLogger(__name__)
        self.prefer_langgraph = prefer_langgraph
        self._load_available_models()
        self._start_cleanup_thread()

    def _load_available_models(self):
        """Load all models from the model service into the available_models dictionary."""
        try:
            models_file = "/app/.models.json"
            if not os.path.exists(models_file):
                self.logger.error(f"Models config file not found: {models_file}")
                return

            with open(models_file, "r", encoding="utf-8") as f:
                models_data = cast(List[Dict[str, Any]], json.load(f))

            for data in models_data:
                try:
                    # Create lora weights and model details
                    loras = [
                        LoraWeight(
                            id=lw.get("id", ""),
                            name=lw.get("name", ""),
                            weight_name=lw.get("weight_name", ""),
                            adapter_name=lw.get("adapter_name", ""),
                            parent_model=lw.get("parent_model", ""),
                        )
                        for lw in data.get("lora_weights", [])
                        if lw
                    ]

                    details_dict = data.get("details", {})
                    if not isinstance(details_dict, dict):
                        details_dict = {}

                    details = ModelDetails(
                        parent_model=str(details_dict.get("parent_model", "")) or None,
                        format=str(details_dict.get("format", "")),
                        family=str(details_dict.get("family", "")),
                        families=(
                            list(details_dict.get("families", []))
                            if isinstance(details_dict.get("families"), list)
                            else []
                        ),
                        parameter_size=str(details_dict.get("parameter_size", "")),
                        quantization_level=str(
                            details_dict.get("quantization_level", "")
                        )
                        or None,
                        specialization=str(details_dict.get("specialization", ""))
                        or None,
                        dtype=str(details_dict.get("dtype", "bf16")),
                        precision=str(details_dict.get("precision", "fp16")),
                        weight=float(details_dict.get("weight", 1.0)),
                        gguf_file=str(details_dict.get("gguf_file", "")) or None,
                        description=str(details_dict.get("description", "")) or None,
                    )

                    model = Model(
                        id=data.get("id"),
                        name=data["name"],
                        model=data["model"],
                        modified_at=data["modified_at"],
                        size=data["size"],
                        digest=data["digest"],
                        pipeline=data.get("pipeline"),
                        lora_weights=loras,
                        details=details,
                        task=data.get("task", "TextToText"),
                    )

                    assert (
                        model.details and model.model
                    ), "Missing required model fields"
                    self._available_models[data["id"]] = model

                except Exception as e:
                    self.logger.error(
                        f"Error creating model from {data.get('id', 'unknown')}: {e}"
                    )

            self.logger.info(f"Loaded {len(self._available_models)} models from config")

        except Exception as e:
            self.logger.error(f"Error loading models config: {e}")

    def get_pipeline[P: str | List[List[float]] | ChatResponse](
        self, profile: ModelProfile, _: Type[P] = ChatResponse
    ) -> BasePipelineCore[P]:
        """
        Get the appropriate pipeline for the given model profile.
        Automatically selects between legacy and modern implementations.
        """
        model_id = profile.model_name

        # Check cache first
        with self._cleanup_lock:
            if model_id in self._pipelines:
                self.logger.info(f"Using cached pipeline for model: {model_id}")
                self._pipelines[model_id].last_accessed = time.time()
                return self._pipelines[model_id].pipeline

        model = self._get_model_by_id(model_id)
        if model is None:
            raise RuntimeError(f"Model with ID '{model_id}' not found.")

        pipe = self.create_pipeline(model, profile)
        if pipe is None:
            raise RuntimeError(f"Failed to create pipeline for model {model.name}.")

        # Cache the pipeline
        with self._cleanup_lock:
            self._pipelines[model_id] = PipelineCacheEntry(pipe)

        return cast(BasePipelineCore[P], pipe)

    def _get_model_by_id(self, model_id: str) -> Model:
        """Retrieve a model by its ID from the available models dictionary."""
        self.logger.info(f"Available models: {list(self._available_models.keys())}")

        if not self._available_models:
            raise RuntimeError("Available models dictionary is empty.")

        if model_id not in self._available_models:
            raise RuntimeError(
                f"Model with ID '{model_id}' not found in available models."
            )

        model = self._available_models[model_id]
        self.logger.info(f"Creating pipeline for model: {model.name} (ID: {model.id})")
        return model

    def create_pipeline(
        self, model: Model, profile: ModelProfile
    ) -> Optional[BasePipelineCore]:
        """
        Enhanced factory method with automatic LangGraph/legacy selection.
        """
        # Clear memory before loading new model
        # hardware_manager.clear_memory()

        if model.task.endswith("TextToText"):

            # Qwen models - prefer LangGraph implementation
            if (
                model.pipeline == "Qwen30A3BQ4KMPipe"
                or model.pipeline == "Qwen30A3BCoderQ4KMPipe"
            ):

                if self.prefer_langgraph:
                    self.logger.info(
                        f"Creating LangGraph Qwen pipeline for model {model.name}"
                    )
                    try:
                        # Import the new LangGraph implementation
                        from .txt2txt.qwen3moe import QwenLangGraphPipe

                        return QwenLangGraphPipe(model, profile)
                    except ImportError as e:
                        self.logger.warning(
                            f"LangGraph implementation not available: {e}"
                        )
                        # Fall back to legacy implementation

                # Legacy implementation fallback
                self.logger.info(
                    f"Creating legacy Qwen pipeline for model {model.name}"
                )
                from .txt2txt.qwen3_a3b import QwenGGUFPipe

                return QwenGGUFPipe(model, profile)

            # Vision-Language models
            if model.pipeline == "Qwen25VLGGUFPipeline":
                if self.prefer_langgraph:
                    self.logger.info(
                        f"Creating LangGraph Qwen 2.5 VL pipeline for model {model.name}"
                    )
                    try:
                        from .imgtxt2txt.qwen25vl import (
                            Qwen25VLLangGraphPipe,
                        )

                        return Qwen25VLLangGraphPipe(model, profile)
                    except ImportError:
                        self.logger.warning(
                            "LangGraph VL implementation not available, using legacy"
                        )

                # Legacy fallback
                self.logger.info(
                    f"Creating legacy Qwen 2.5 VL pipeline for model {model.name}"
                )
                from .imgtxt2txt.qwen25_vl import Qwen25VLGGUFPipe

                return Qwen25VLGGUFPipe(model, profile)

            # BART Summarization
            if model.pipeline == "BARTSummarizationPipe":
                if self.prefer_langgraph:
                    try:
                        from .txt2txt.bartsumm import (
                            BARTSummarizationLangGraphPipe,
                        )

                        return BARTSummarizationLangGraphPipe(model, profile)
                    except ImportError:
                        pass

                from .txt2txt.bartsum import BARTSummarizationPipe

                return BARTSummarizationPipe(model, profile)

        # Image generation tasks
        if model.task == "TextToImage":
            if model.pipeline == "FluxPipeline":
                self.logger.info(f"Creating Flux pipeline for model {model.name}")
                from .txt2img.flux import FluxPipe

                return FluxPipe(model, profile)

        # Image-to-image tasks
        if model.task == "ImageToImage":
            if model.pipeline == "FluxKontextPipeline":
                self.logger.info(
                    f"Creating FluxKontext pipeline for model {model.name}"
                )
                from .img2img.flux import FluxKontextPipe

                return FluxKontextPipe(model, profile)

        # Embedding tasks
        if model.task == "TextToEmbeddings":
            if model.pipeline == "NomicEmbedTextPipe":
                self.logger.info(
                    f"Creating Nomic Embed Text pipeline for model {model.name}"
                )
                try:
                    from .emb.nom2 import NomicEmbedTextPipe

                    return NomicEmbedTextPipe(model, profile)
                except Exception as e:
                    self.logger.error(f"Failed to initialize NomicEmbedTextPipe: {e}")
                    raise

            if model.pipeline == "Qwen3EmbeddingPipe":
                self.logger.info(
                    f"Creating Qwen3 Embedding pipeline for model {model.name}"
                )
                try:
                    from .emb.qwen3emb import Qwen3EmbeddingPipe

                    return Qwen3EmbeddingPipe(model, profile)
                except Exception as e:
                    self.logger.error(f"Failed to initialize Qwen3EmbeddingPipe: {e}")
                    raise

        # Reranking tasks
        if model.task == "TextToRanking":
            if model.pipeline == "Qwen3RerankerPipe":
                self.logger.info(
                    f"Creating Qwen3 Reranker pipeline for model {model.name}"
                )
                try:
                    from .emb.qwen3rr import Qwen3RerankerPipe

                    return Qwen3RerankerPipe(model, profile)
                except Exception as e:
                    self.logger.error(f"Failed to initialize Qwen3RerankerPipe: {e}")
                    raise

        self.logger.error(
            f"Unsupported pipeline type '{model.pipeline}' for model {model.name}"
        )
        return None

    def _start_cleanup_thread(self):
        """Start a background thread to periodically clean up expired cache entries."""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_cache_task, daemon=True
            )
            self._cleanup_thread.start()

    def _cleanup_cache_task(self):
        """Background task that periodically checks and removes expired cache entries."""
        self.logger.info("Starting pipeline cache cleanup thread")

        while True:
            time.sleep(60)  # Check every minute
            try:
                self._cleanup_expired_entries()
            except Exception as e:
                self.logger.error(f"Error in cache cleanup task: {str(e)}")

    def _cleanup_expired_entries(self) -> None:
        """Remove expired entries from the pipeline cache."""
        current_time = time.time()
        models_to_remove = []

        with self._cleanup_lock:
            # Find expired entries
            for model_id, entry in self._pipelines.items():
                if current_time - entry.last_accessed > self._cache_timeout:
                    models_to_remove.append(model_id)

            # Remove expired entries
            for model_id in models_to_remove:
                self.logger.info(
                    f"Removing expired pipeline for model {model_id} from cache"
                )
                pipe_entry = self._pipelines.pop(model_id, None)
                if pipe_entry and pipe_entry.pipeline:
                    self._cleanup_pipeline_resources(pipe_entry.pipeline)

    def _cleanup_pipeline_resources(self, pipeline: BasePipelineCore) -> None:
        """Clean up resources used by a pipeline instance."""
        try:
            if pipeline is not None:
                self.logger.debug(f"Cleaning up {type(pipeline).__name__}")
                del pipeline
        except Exception:
            self.logger.warning("Unexpected error during pipeline cleanup")

    def clear_cache(self, model_id: Optional[str] = None) -> None:
        """Manually clear the pipeline cache and clean up resources."""
        with self._cleanup_lock:
            if model_id is not None:
                self.logger.info(
                    f"Manually removing pipeline for model {model_id} from cache"
                )
                pipe_entry = self._pipelines.pop(model_id, None)
                if pipe_entry and pipe_entry.pipeline:
                    self._cleanup_pipeline_resources(pipe_entry.pipeline)
            else:
                self.logger.info("Manually clearing all pipelines from cache")
                model_ids = list(self._pipelines.keys())
                for m_id in model_ids:
                    pipe_entry = self._pipelines.pop(m_id, None)
                    if pipe_entry and pipe_entry.pipeline:
                        self._cleanup_pipeline_resources(pipe_entry.pipeline)


# Create factory instance with LangGraph preference
pipeline_factory = ModernPipelineFactory(prefer_langgraph=True)
