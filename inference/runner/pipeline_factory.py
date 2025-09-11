"""
Production-ready pipeline factory with weakref caching, background cleanup, and
modern/legacy pipeline selection. Replaces the previous garbled version.
"""

import json
import logging
import os
import time
import threading
from typing import Any, Dict, List, Optional, Type, cast, TypeVar
from contextlib import contextmanager
from enum import IntEnum

from .pipeline_cache import (
    LocalPipelineCacheManager,
    PipelinePriority as LocalCachePriority,
)

from models import (
    Model,
    LoraWeight,
    ModelDetails,
    ModelProfile,
    ChatResponse,
    ModelProvider,
)
from .pipelines.base import BasePipelineCore, PipeReturn

from .pipelines.llamacpp.base_llamacpp import BaseLlamaCppPipeline
from utils.hardware_manager import hardware_manager


class PipelinePriority(IntEnum):
    """Pipeline priority levels for cache eviction."""

    LOW = 1  # Tool generation, etc. (evict first)
    MEDIUM = 5  # Standard pipelines
    NORMAL = 5  # Standard pipelines
    HIGH = 10  # Critical pipelines (main chat models)
    CRITICAL = 20  # For primary/main models that should rarely be evicted


T = TypeVar("T", bound=PipeReturn)


class PipelineFactory:
    """
    Factory that:
    - Loads model metadata from /app/.models.json
    - Creates appropriate pipeline implementation per model.task/pipeline
    - Caches pipelines with weakrefs and cleans them up after inactivity
    """

    _available_models: Dict[str, Model] = {}
    _cache_timeout = 300  # seconds
    _cleanup_lock = threading.RLock()

    def __init__(self):
        """Initialize production-ready pipeline factory with lifecycle management."""
        self.logger = logging.getLogger(__name__)

        # Available models (populate from config)
        self._available_models: Dict[str, Model] = {}

        # Use our new local pipeline cache
        self.local_cache = LocalPipelineCacheManager()

        # Coordination for memory-constrained loading
        self._coord_lock = threading.Lock()
        self._coord_cond = threading.Condition(self._coord_lock)
        self._active_loads = 0  # Number of concurrent pipeline loads
        self._active_local_uses = 0  # Number of active local pipeline uses
        self._loading_models: Dict[str, threading.Event] = (
            {}
        )  # Track which models are currently loading

        # Load configurations
        self.prefer_langgraph = True  # default - can be controlled
        self._load_available_models()

    # Legacy cleanup thread no longer needed (handled in LocalPipelineCacheManager)

    # ---------- Model loading ----------

    def _load_available_models(self) -> None:
        try:
            models_file = "/app/.models.json"
            if not os.path.exists(models_file):
                # Fallback to env-specified path for local/dev testing
                env_path = os.environ.get("MODELS_FILE_PATH")
                if env_path and os.path.exists(env_path):
                    models_file = env_path
                    self.logger.info(
                        f"Using models config from MODELS_FILE_PATH: {models_file}"
                    )
                else:
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

    def _estimate_pipeline_memory_requirements(self, model: Model) -> float:
        """
        Estimate memory requirements for a pipeline based on model characteristics.
        Returns estimated bytes required.

        Uses parameter-based calculation as primary method, with size as fallback validation.
        """
        base_memory = 512 * 1024 * 1024  # 512MB base overhead
        model_size_bytes = 0

        # Primary: Calculate from parameter size (most reliable)
        if (
            hasattr(model, "details")
            and model.details
            and hasattr(model.details, "parameter_size")
        ):
            param_size = model.details.parameter_size
            if param_size:
                try:
                    # Parse parameter size (e.g., "30B", "20B", "3B", "475M")
                    param_str = param_size.upper().strip()

                    if param_str.endswith("B"):
                        params = float(param_str[:-1]) * 1_000_000_000
                    elif param_str.endswith("M"):
                        params = float(param_str[:-1]) * 1_000_000
                    elif param_str.endswith("K"):
                        params = float(param_str[:-1]) * 1_000
                    else:
                        # Assume it's in billions if no suffix and > 1
                        num_val = float(param_str)
                        params = (
                            num_val * 1_000_000_000
                            if num_val > 1
                            else num_val * 1_000_000
                        )

                    # Estimate bytes per parameter based on quantization
                    quantization = getattr(
                        model.details, "quantization_level", "q4"
                    ).lower()
                    if "q4" in quantization or "iq4" in quantization:
                        bytes_per_param = 0.5  # 4-bit
                    elif "q5" in quantization:
                        bytes_per_param = 0.625  # 5-bit
                    elif "q8" in quantization:
                        bytes_per_param = 1.0  # 8-bit
                    elif any(x in quantization for x in ["fp16", "bf16", "f16"]):
                        bytes_per_param = 2.0  # 16-bit
                    else:
                        bytes_per_param = 4.0  # fp32 default

                    model_size_bytes = int(params * bytes_per_param)

                    self.logger.debug(
                        f"Parameter-based size for {model.name}: {param_size} = {params:,.0f} params "
                        f"* {bytes_per_param} bytes/param = {model_size_bytes/1e9:.2f}GB"
                    )

                except (ValueError, AttributeError) as e:
                    self.logger.warning(
                        f"Could not parse parameter size '{param_size}': {e}"
                    )

        # Fallback: Use model.size if parameter calculation failed and size seems reasonable
        if model_size_bytes == 0 and hasattr(model, "size") and model.size:
            # Use model.size as fallback, but validate it's reasonable
            if model.size < 100 * 1024 * 1024 * 1024:  # Less than 100GB
                model_size_bytes = model.size
                self.logger.debug(f"Using model.size: {model_size_bytes/1e9:.2f}GB")
            else:
                self.logger.warning(
                    f"Model size {model.size/1e9:.2f}GB seems unreasonable, using task-based estimate"
                )

        # Last resort: Task-based defaults (more conservative)
        if model_size_bytes == 0:
            task = getattr(model, "task", "TextToText")
            if task == "TextToEmbeddings":
                model_size_bytes = 1 * 1024 * 1024 * 1024  # 1GB
            elif task in ["TextToText", "VisionTextToText"]:
                model_size_bytes = 4 * 1024 * 1024 * 1024  # 4GB (reduced from 8GB)
            elif task in ["TextToImage", "ImageToImage"]:
                model_size_bytes = 8 * 1024 * 1024 * 1024  # 8GB (reduced from 12GB)
            else:
                model_size_bytes = 2 * 1024 * 1024 * 1024  # 2GB (reduced from 4GB)

            self.logger.debug(
                f"Using task-based default for {task}: {model_size_bytes/1e9:.2f}GB"
            )

        # Conservative context memory estimation
        context_memory = 0
        task = getattr(model, "task", "")
        if task in ["TextToText", "VisionTextToText"]:
            # Context memory scales with model size but caps at 2GB
            context_memory = min(model_size_bytes * 0.1, 2 * 1024 * 1024 * 1024)

        total_estimated = base_memory + model_size_bytes + context_memory
        # Add safety margin for local providers due to KV cache fragmentation / extra overhead
        provider = getattr(model, "provider", "") or ""
        if provider in {ModelProvider.LLAMA_CPP, ModelProvider.STABLE_DIFFUSION_CPP}:
            total_estimated *= 1.10

        self.logger.info(
            f"Memory estimate for {model.name}: "
            f"Model: {model_size_bytes/1e9:.2f}GB + "
            f"Context: {context_memory/1e9:.2f}GB + "
            f"Base: {base_memory/1e6:.0f}MB = "
            f"Total: {total_estimated/1e9:.2f}GB"
        )

        return total_estimated

    def get_pipeline(
        self,
        profile: ModelProfile,
        expected_type: Type[T],
        priority: PipelinePriority = PipelinePriority.NORMAL,
    ) -> BasePipelineCore[T]:
        model_id = profile.model_name
        model = self._get_model_by_id(model_id)
        if not model:
            raise RuntimeError(f"Model with ID '{model_id}' not found.")

        # Local providers -> managed cached path
        if getattr(model, "provider", None) in {
            ModelProvider.LLAMA_CPP,
            ModelProvider.STABLE_DIFFUSION_CPP,
        }:
            # Use a factory function that handles coordination internally
            def create_with_coordination(
                m: Model, p: ModelProfile, e: Optional[Type[PipeReturn]]
            ) -> Optional[BasePipelineCore]:
                required_estimate = self._estimate_pipeline_memory_requirements(m)
                self._acquire_load_slot(required_estimate, model_id)
                try:
                    return self.create_pipeline(m, p, expected_type=e)
                finally:
                    self._release_load_slot(model_id)

            pipeline = self.local_cache.get_or_create(
                model,
                profile,
                expected_type,
                LocalCachePriority(priority.value),  # map enum values
                create_with_coordination,
            )
            try:
                if isinstance(pipeline, BaseLlamaCppPipeline):  # type: ignore
                    self.logger.debug(
                        f"Pipeline {model.name} is llama.cpp based (local cached)"
                    )
            except Exception:
                pass
            return cast(BasePipelineCore[T], pipeline)

        # Remote / API providers -> create transient each call, no caching
        pipeline = self.create_pipeline(model, profile, expected_type=expected_type)
        if not pipeline:
            raise RuntimeError(
                f"Failed to create pipeline for remote model {getattr(model, 'name', model_id)}."
            )
        self.logger.debug(
            f"Created transient pipeline for remote provider {getattr(model, 'provider', 'unknown')} ({model.name})"
        )
        return cast(BasePipelineCore[T], pipeline)

    def clear_cache(self, model_id: Optional[str] = None) -> None:
        """Delegate to local cache manager (only impacts local models)."""
        self.local_cache.clear_cache(model_id)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return combined stats (local cache + model availability + hardware)."""
        local_stats = self.local_cache.stats()
        memory_stats = hardware_manager.update_all_memory_stats()
        return {
            "local_cache": local_stats,
            "available_models": len(self._available_models),
            "cache_timeout": self._cache_timeout,
            "memory_stats": {
                device_id: {
                    "total_mb": stats.mem_total,
                    "used_mb": stats.mem_used,
                    "free_mb": stats.mem_free,
                    "utilization_percent": stats.mem_util,
                    "gpu_utilization_percent": stats.gpu_util,
                    "temperature_c": stats.temperature,
                }
                for device_id, stats in memory_stats.items()
            },
            "hardware_manager_available": hardware_manager.has_gpu,
            "gpu_count": hardware_manager.gpu_count if hardware_manager.has_gpu else 0,
        }

    @contextmanager
    def pipeline(
        self,
        profile: ModelProfile,
        t: Type[T],
        priority: PipelinePriority = PipelinePriority.NORMAL,
    ):
        pipeline = self.get_pipeline(profile, t, priority)
        is_local = False
        try:
            provider = getattr(pipeline.model, "provider", None)  # type: ignore[attr-defined]
            if provider in {
                ModelProvider.LLAMA_CPP,
                ModelProvider.STABLE_DIFFUSION_CPP,
            }:
                is_local = True
        except Exception:
            pass
        if is_local:
            with self._coord_cond:
                self._active_local_uses += 1
        try:
            yield pipeline
        finally:
            if is_local:
                # self.local_cache.release(pipeline) # release is not a method on the cache manager
                with self._coord_cond:
                    self._active_local_uses = max(0, self._active_local_uses - 1)
                    self._coord_cond.notify_all()

    # ---------- Coordination Helpers ----------

    def _acquire_load_slot(
        self, required_bytes: float, model_id: str, timeout: float = 300.0
    ) -> None:
        """Serialize local pipeline loads when memory is constrained.

        Waits until either memory is available AND no other load in progress,
        or raises RuntimeError after timeout if still impossible.

        This method prevents concurrent loads but allows checking memory for the same operation.
        If the same model is already being loaded, waits for that specific load to complete.
        """
        start = time.time()

        # Check if this specific model is already being loaded
        with self._coord_cond:
            if model_id in self._loading_models:
                self.logger.info(
                    f"Model {model_id} is already being loaded, waiting for completion..."
                )
                # Wait for the existing load to complete
                load_event = self._loading_models[model_id]
                # Release condition lock while waiting for the event
                self._coord_cond.release()
                try:
                    if not load_event.wait(timeout=timeout):
                        raise RuntimeError(
                            f"Timeout waiting for {model_id} to finish loading"
                        )
                finally:
                    self._coord_cond.acquire()
                return

            # Mark this model as being loaded
            self._loading_models[model_id] = threading.Event()

        # Initial check
        mem_ok = hardware_manager.check_memory_available(required_bytes)
        if not mem_ok:
            self.logger.warning(
                f"Initial memory check failed (need {required_bytes/1e9:.2f}GB). Forcing cache cleanup and retrying."
            )
            self.force_memory_cleanup()
            mem_ok = hardware_manager.check_memory_available(required_bytes)

        try:
            with self._coord_cond:
                while True:
                    if mem_ok and self._active_loads == 0:
                        self._active_loads += 1
                        return
                    # If memory not ok but there are active uses or active load, wait
                    if (not mem_ok) and (
                        self._active_local_uses > 0 or self._active_loads > 0
                    ):
                        remaining = timeout - (time.time() - start)
                        if remaining <= 0:
                            raise RuntimeError(
                                f"Timeout waiting for memory to load pipeline (need {required_bytes/1e9:.2f}GB)"
                            )
                        self._coord_cond.wait(timeout=remaining)
                        # After waiting, re-check memory and potentially force cleanup again
                        mem_ok = hardware_manager.check_memory_available(required_bytes)
                        if not mem_ok:
                            self.force_memory_cleanup()
                            mem_ok = hardware_manager.check_memory_available(
                                required_bytes
                            )
                        continue
                    # No other active load and memory still insufficient -> abort
                    if not mem_ok:
                        raise RuntimeError(
                            f"Insufficient memory to load pipeline (need {required_bytes/1e9:.2f}GB)"
                        )
                    # Else some other load in progress
                    remaining = timeout - (time.time() - start)
                    if remaining <= 0:
                        raise RuntimeError(
                            f"Timeout waiting for prior pipeline load to finish (need {required_bytes/1e9:.2f}GB)"
                        )
                    self._coord_cond.wait(timeout=remaining)
        except Exception:
            # Clean up on error
            with self._coord_cond:
                if model_id in self._loading_models:
                    self._loading_models[
                        model_id
                    ].set()  # Signal completion even on error
                    del self._loading_models[model_id]
            raise

    def _release_load_slot(self, model_id: str) -> None:
        with self._coord_cond:
            if self._active_loads > 0:
                self._active_loads -= 1
            # Mark the model as loaded and clean up
            if model_id in self._loading_models:
                self._loading_models[model_id].set()  # Signal completion
                del self._loading_models[model_id]
            self._coord_cond.notify_all()

    def set_pipeline_priority(self, model_id: str, priority: PipelinePriority) -> bool:
        # Map external priority enum to local cache priority and delegate
        success = self.local_cache.set_priority(
            model_id, LocalCachePriority(priority.value)
        )
        if success:
            self.logger.info(
                f"Updated pipeline priority for {model_id} -> {priority.name}"
            )
        return success

    def get_pipeline_info(self) -> Dict[str, Dict[str, Any]]:
        stats = self.local_cache.stats()
        entries = stats.get("entries", {})
        transformed: Dict[str, Dict[str, Any]] = {}
        now = time.time()
        for mid, data in entries.items():
            last = data.get("last_accessed", now)
            transformed[mid] = {
                "priority": data.get("priority"),
                "age_minutes": round((now - last) / 60.0, 1),
                "access_count": data.get("access_count"),
            }
        return transformed

    def force_memory_cleanup(self) -> int:
        # Delegate to local cache full cleanup; ignore target bytes (future improvement)
        evicted = self.local_cache.force_cleanup()
        self.logger.info(f"Force cleanup evicted {evicted} local pipelines")
        return evicted

    # Backward compatibility wrapper (legacy code may call this)
    def force_resource_cleanup(
        self, _target_free_memory_gb: float = 1.0
    ) -> int:  # noqa: ARG002
        """Alias for force_memory_cleanup kept for older callers."""
        return self.force_memory_cleanup()

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
        self,
        model: Model,
        profile: ModelProfile,
        expected_type: Optional[Type[PipeReturn]] = None,
    ) -> Optional[BasePipelineCore]:
        try:
            self.logger.info(f"Creating pipeline for {model.name} (task: {model.task})")
            if model.task.endswith("TextToText"):
                return self._create_text_pipeline(model, profile, expected_type)
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

            return None

    def _create_text_pipeline(
        self,
        model: Model,
        profile: ModelProfile,
        expected_type: Optional[Type[PipeReturn]] = None,
    ) -> Optional[BasePipelineCore]:
        self.logger.info(
            f"Creating text pipeline for model: {model.name}, pipeline: {model.pipeline}"
        )
        if model.pipeline in ("Qwen30A3BQ4KMPipe", "Qwen30A3BCoderQ4KMPipe"):
            self.logger.info(
                f"Creating Qwen pipeline, prefer_langgraph={self.prefer_langgraph}"
            )
            from .pipelines.txt2txt.qwen3moe import QwenLangGraphPipe

            self.logger.info("Attempting to create QwenLangGraphPipe v2")
            try:
                # Try with expected_return_type first (preferred)
                pipeline = QwenLangGraphPipe(
                    model, profile, expected_return_type=expected_type
                )
            except TypeError as e:
                if "unexpected keyword argument" in str(e):
                    self.logger.warning(
                        f"QwenLangGraphPipe doesn't accept expected_return_type, falling back: {e}"
                    )
                    # Fallback for older signature
                    pipeline = QwenLangGraphPipe(model, profile)
                else:
                    raise
            self.logger.info("Successfully created QwenLangGraphPipe v2")
            return pipeline

        if model.pipeline == "Qwen25VLGGUFPipeline":
            # File may not exist; fallback handled below
            from .pipelines.imgtxt2txt.qwen25_vl import Qwen25VLPipeline

            return Qwen25VLPipeline(model, profile)

        if model.pipeline == "LlamaChatSummPipe":
            from .pipelines.txt2txt.llamachatsum import LlamaChatSummPipe

            return LlamaChatSummPipe(
                model, profile, return_type=expected_type or ChatResponse
            )

        if model.pipeline == "OpenAiGptOssPipe":
            from .pipelines.txt2txt.openai_gpt_oss import OpenAiGptOssPipe

            return OpenAiGptOssPipe(model, profile, expected_return_type=expected_type)

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

    # (Removed duplicate legacy cleanup method; single alias earlier in file)


# Create global factory instance
pipeline_factory = PipelineFactory()
