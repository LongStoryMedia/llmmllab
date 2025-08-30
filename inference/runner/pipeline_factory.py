"""
Production-ready pipeline factory with weakref caching, background cleanup, and
modern/legacy pipeline selection. Replaces the previous garbled version.
"""

import json
import logging
import os
import time
import threading
from typing import Any, Dict, List, Optional, Type, cast
from contextlib import contextmanager
import weakref

from models import Model, LoraWeight, ModelDetails, ModelProfile, ChatResponse
from .pipelines.base import BasePipelineCore, PipeReturn


class PipelineCacheEntry:
    """Cache entry with automatic cleanup via weak references."""

    def __init__(self, pipeline: BasePipelineCore, timestamp: Optional[float] = None):
        self._pipeline_ref = weakref.ref(pipeline)
        self.last_accessed = timestamp if timestamp is not None else time.time()
        self.creation_time = time.time()

    @property
    def pipeline(self) -> Optional[BasePipelineCore]:
        """Return the live pipeline instance, or None if it has been GC'd."""
        return self._pipeline_ref()

    def is_alive(self) -> bool:
        return self._pipeline_ref() is not None

    def touch(self) -> None:
        self.last_accessed = time.time()


class PipelineFactory:
    """
    Factory that:
    - Loads model metadata from /app/.models.json
    - Creates appropriate pipeline implementation per model.task/pipeline
    - Caches pipelines with weakrefs and cleans them up after inactivity
    """

    _pipelines: Dict[str, PipelineCacheEntry] = {}
    _available_models: Dict[str, Model] = {}
    _cache_timeout = 300  # seconds
    _cleanup_thread = None
    _cleanup_lock = threading.RLock()

    def __init__(self, prefer_langgraph: bool = True, cache_timeout: int = 300):
        self.logger = logging.getLogger(__name__)
        self.prefer_langgraph = prefer_langgraph
        self._cache_timeout = cache_timeout

        self._load_available_models()
        self._start_cleanup_thread()

    # ---------- Model loading ----------

    def _load_available_models(self) -> None:
        try:
            models_file = "/app/.models.json"
            if not os.path.exists(models_file):
                self.logger.error(f"Models config file not found: {models_file}")
                return

            with open(models_file, "r", encoding="utf-8") as f:
                models_data = json.load(f)

            if not isinstance(models_data, list):
                self.logger.error("Models config is not a list; ignoring")
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
                f"Loaded {loaded_count}/{len(models_data)} models from config"
            )

        except Exception as e:
            self.logger.error(f"Error loading models config: {e}")

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
            details = ModelDetails(
                parent_model=details_dict.get("parent_model"),
                format=str(details_dict.get("format", "")),
                family=str(details_dict.get("family", "")),
                families=list(details_dict.get("families", [])),
                parameter_size=str(details_dict.get("parameter_size", "")),
                quantization_level=details_dict.get("quantization_level"),
                specialization=details_dict.get("specialization"),
                dtype=str(details_dict.get("dtype", "bf16")),
                precision=str(details_dict.get("precision", "fp16")),
                weight=float(details_dict.get("weight", 1.0)),
                gguf_file=details_dict.get("gguf_file"),
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

    def get_pipeline[T: PipeReturn](
        self, profile: ModelProfile, _: Type[T]
    ) -> BasePipelineCore[T]:
        model_id = profile.model_name

        # Cache lookup
        with self._cleanup_lock:
            entry = self._pipelines.get(model_id)
            if entry and entry.is_alive():
                entry.touch()
                pipeline = entry.pipeline
                if pipeline:
                    self.logger.debug(f"Using cached pipeline for model: {model_id}")
                    return pipeline
            elif entry:
                self._pipelines.pop(model_id, None)

        # Build fresh pipeline
        model = self._get_model_by_id(model_id)
        if not model:
            raise RuntimeError(f"Model with ID '{model_id}' not found.")

        pipeline = self.create_pipeline(model, profile)
        if not pipeline:
            raise RuntimeError(
                f"Failed to create pipeline for model {getattr(model, 'name', model_id)}."
            )

        with self._cleanup_lock:
            self._pipelines[model_id] = PipelineCacheEntry(pipeline)

        self.logger.info(f"Created and cached pipeline for model: {model.name}")
        return cast(BasePipelineCore[T], pipeline)

    def clear_cache(self, model_id: Optional[str] = None) -> None:
        with self._cleanup_lock:
            if model_id is not None:
                entry = self._pipelines.pop(model_id, None)
                if entry and entry.pipeline:
                    self._cleanup_pipeline_resources(entry.pipeline)
                self.logger.info(f"Cleared cache for model {model_id}")
            else:
                for m_id, entry in list(self._pipelines.items()):
                    self._pipelines.pop(m_id, None)
                    if entry and entry.pipeline:
                        self._cleanup_pipeline_resources(entry.pipeline)
                self.logger.info("Cleared all pipeline cache entries")

    def get_cache_stats(self) -> Dict[str, Any]:
        with self._cleanup_lock:
            alive_count = sum(
                1 for entry in self._pipelines.values() if entry.is_alive()
            )
            return {
                "total_entries": len(self._pipelines),
                "alive_entries": alive_count,
                "dead_entries": len(self._pipelines) - alive_count,
                "available_models": len(self._available_models),
                "cache_timeout": self._cache_timeout,
            }

    @contextmanager
    def pipeline[T: PipeReturn](self, profile: ModelProfile, t: Type[T]):
        pipeline = self.get_pipeline(profile, t)
        try:
            yield pipeline
        finally:
            # Keep cached; periodic cleanup will handle expiry
            pass

    # ---------- Internals ----------

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
        self, model: Model, profile: ModelProfile
    ) -> Optional[BasePipelineCore]:
        try:
            self.logger.info(f"Creating pipeline for {model.name} (task: {model.task})")
            if model.task.endswith("TextToText"):
                return self._create_text_pipeline(model, profile)
            if model.task == "TextToEmbeddings":
                return self._create_embedding_pipeline(model, profile)
            if model.task == "TextToRanking":
                return self._create_reranking_pipeline(model, profile)
            if model.task == "TextToImage":
                return self._create_image_pipeline(model, profile)
            if model.task == "ImageToImage":
                return self._create_image_to_image_pipeline(model, profile)
            self.logger.error(f"Unsupported task type: {model.task}")
            return None
        except Exception as e:
            self.logger.error(f"Error creating pipeline for {model.name}: {e}")
            return None

    def _create_text_pipeline(
        self, model: Model, profile: ModelProfile
    ) -> Optional[BasePipelineCore]:
        if model.pipeline in ("Qwen30A3BQ4KMPipe", "Qwen30A3BCoderQ4KMPipe"):
            if self.prefer_langgraph:
                try:
                    from .pipelines.txt2txt.qwen3moe import QwenLangGraphPipe

                    return QwenLangGraphPipe(model, profile, return_type=ChatResponse)
                except ImportError as e:
                    self.logger.warning(f"LangGraph implementation not available: {e}")
            # The GGUF text pipeline is currently disabled/commented out
            return None

        if model.pipeline == "Qwen25VLGGUFPipeline":
            if self.prefer_langgraph:
                try:
                    # File may not exist; fallback handled below
                    from .pipelines.imgtxt2txt.qwen25vl import Qwen25VLLangGraphPipe  # type: ignore

                    return Qwen25VLLangGraphPipe(model, profile)
                except ImportError:
                    self.logger.warning("VL LangGraph implementation not available")
            from .pipelines.imgtxt2txt.qwen25_vl import Qwen25VLGGUFPipe

            return Qwen25VLGGUFPipe(model, profile)

        if model.pipeline == "BARTSummarizationPipe":
            if self.prefer_langgraph:
                try:
                    # File may not exist; fallback handled below
                    from .pipelines.txt2txt.bartsumm import BARTSummarizationLangGraphPipe  # type: ignore

                    return BARTSummarizationLangGraphPipe(model, profile)
                except ImportError:
                    pass
            from .pipelines.txt2txt.bartsum import BARTSummarizationPipe

            return BARTSummarizationPipe(model, profile, return_type=ChatResponse)

        return None

    def _create_embedding_pipeline(
        self, model: Model, profile: ModelProfile
    ) -> Optional[BasePipelineCore]:
        if model.pipeline == "NomicEmbedTextPipe":
            try:
                from .pipelines.emb.nom2 import NomicEmbedTextPipe

                return NomicEmbedTextPipe(model, profile)
            except Exception as e:
                self.logger.error(f"Failed to initialize NomicEmbedTextPipe: {e}")
                return None
        if model.pipeline == "Qwen3EmbeddingPipe":
            try:
                from .pipelines.emb.qwen3emb import Qwen3EmbeddingPipe

                return Qwen3EmbeddingPipe(model, profile)
            except Exception as e:
                self.logger.error(f"Failed to initialize Qwen3EmbeddingPipe: {e}")
                return None
        return None

    def _create_reranking_pipeline(
        self, model: Model, _profile: ModelProfile
    ) -> Optional[BasePipelineCore]:
        """Create reranking pipeline (currently unavailable)."""
        if model.pipeline == "Qwen3RerankerPipe":
            # Reranker implementation is currently commented out / unavailable
            self.logger.warning(
                "Qwen3RerankerPipe is not available; skipping reranking pipeline creation"
            )
            return None
        return None

    def _create_image_pipeline(
        self, model: Model, profile: ModelProfile
    ) -> Optional[BasePipelineCore]:
        if model.pipeline == "FluxPipeline":
            try:
                from .pipelines.txt2img.flux import FluxPipe

                return FluxPipe(model, profile)
            except Exception as e:
                self.logger.error(f"Failed to initialize FluxPipe: {e}")
                return None
        return None

    def _create_image_to_image_pipeline(
        self, model: Model, profile: ModelProfile
    ) -> Optional[BasePipelineCore]:
        if model.pipeline == "FluxKontextPipeline":
            try:
                from .pipelines.img2img.flux import FluxKontextPipe

                return FluxKontextPipe(model, profile)
            except Exception as e:
                self.logger.error(f"Failed to initialize FluxKontextPipe: {e}")
                return None
        return None

    # ---------- Cache and cleanup ----------

    def _start_cleanup_thread(self) -> None:
        if (
            self._cleanup_thread is None
            or not getattr(self._cleanup_thread, "is_alive", lambda: False)()
        ):
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_task, daemon=True, name="PipelineCleanup"
            )
            self._cleanup_thread.start()
            self.logger.info("Started pipeline cleanup thread")

    def _cleanup_task(self) -> None:
        self.logger.info("Pipeline cleanup thread started")
        while True:
            try:
                time.sleep(60)
                self._cleanup_expired_entries()
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")

    def _cleanup_expired_entries(self) -> None:
        current_time = time.time()
        expired_keys: List[str] = []
        with self._cleanup_lock:
            for model_id, entry in self._pipelines.items():
                if (
                    current_time - entry.last_accessed > self._cache_timeout
                ) or not entry.is_alive():
                    expired_keys.append(model_id)
            for model_id in expired_keys:
                entry = self._pipelines.pop(model_id, None)
                if entry:
                    pipeline = entry.pipeline
                    if pipeline:
                        self._cleanup_pipeline_resources(pipeline)
                    self.logger.info(f"Removed expired pipeline for model {model_id}")

    def _cleanup_pipeline_resources(self, pipeline: BasePipelineCore) -> None:
        try:
            cleanup_fn = getattr(pipeline, "cleanup", None)
            if callable(cleanup_fn):
                cleanup_fn()
            llm = getattr(pipeline, "llm", None)
            if llm is not None:
                llm_cleanup = getattr(llm, "cleanup", None)
                if callable(llm_cleanup):
                    llm_cleanup()
            self.logger.debug(f"Cleaned up {type(pipeline).__name__}")
        except Exception as e:
            self.logger.error(f"Error cleaning up pipeline: {e}")


# Create global factory instance
pipeline_factory = PipelineFactory(prefer_langgraph=True)
