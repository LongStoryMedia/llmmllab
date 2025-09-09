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
from utils.hardware_manager import hardware_manager


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

        # Last resort: Task-based defaults
        if model_size_bytes == 0:
            task = getattr(model, "task", "TextToText")
            if task == "TextToEmbeddings":
                model_size_bytes = 1 * 1024 * 1024 * 1024  # 1GB
            elif task in ["TextToText", "VisionTextToText"]:
                model_size_bytes = 8 * 1024 * 1024 * 1024  # 8GB
            elif task in ["TextToImage", "ImageToImage"]:
                model_size_bytes = 12 * 1024 * 1024 * 1024  # 12GB
            else:
                model_size_bytes = 4 * 1024 * 1024 * 1024  # 4GB

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

        self.logger.info(
            f"Memory estimate for {model.name}: "
            f"Model: {model_size_bytes/1e9:.2f}GB + "
            f"Context: {context_memory/1e9:.2f}GB + "
            f"Base: {base_memory/1e6:.0f}MB = "
            f"Total: {total_estimated/1e9:.2f}GB"
        )

        return total_estimated

    def _ensure_sufficient_memory(
        self, required_bytes: float, exclude_model: Optional[str] = None
    ) -> bool:
        """
        Ensure sufficient memory is available by evicting cached pipelines if necessary.

        Args:
            required_bytes: Memory required in bytes
            exclude_model: Model ID to exclude from eviction (e.g., the one being created)

        Returns:
            True if sufficient memory is available, False if not possible
        """
        # First check if memory is already available
        if hardware_manager.check_memory_available(required_bytes):
            return True

        self.logger.info(
            f"Insufficient memory for pipeline requiring {required_bytes/1e9:.2f}GB. "
            "Attempting cache eviction..."
        )

        # Clear any already-dead entries first
        with self._cleanup_lock:
            dead_keys = [
                k for k, entry in self._pipelines.items() if not entry.is_alive()
            ]
            for k in dead_keys:
                self._pipelines.pop(k, None)

        # Try gentle memory clearing first
        hardware_manager.clear_memory(aggressive=False)
        if hardware_manager.check_memory_available(required_bytes):
            self.logger.info("Sufficient memory available after gentle cleanup")
            return True

        # Get entries sorted by last access time (oldest first)
        with self._cleanup_lock:
            eviction_candidates = [
                (model_id, entry)
                for model_id, entry in self._pipelines.items()
                if entry.is_alive() and model_id != exclude_model
            ]
            eviction_candidates.sort(key=lambda x: x[1].last_accessed)

        # Evict cached pipelines until we have enough memory
        evicted_count = 0
        for model_id, entry in eviction_candidates:
            with self._cleanup_lock:
                removed_entry = self._pipelines.pop(model_id, None)
                if removed_entry and removed_entry.pipeline:
                    self._cleanup_pipeline_resources(removed_entry.pipeline)
                evicted_count += 1

            self.logger.info(f"Evicted cached pipeline for model: {model_id}")

            # Try more aggressive memory clearing after each eviction
            hardware_manager.clear_memory(aggressive=True)

            if hardware_manager.check_memory_available(required_bytes):
                self.logger.info(
                    f"Sufficient memory available after evicting {evicted_count} pipelines"
                )
                return True

        # Last resort: nuclear memory clearing
        self.logger.warning("Attempting nuclear memory clearing as last resort")
        hardware_manager.clear_memory(aggressive=True, nuclear=False)

        if hardware_manager.check_memory_available(required_bytes):
            self.logger.info("Sufficient memory available after nuclear cleanup")
            return True

        self.logger.error(
            f"Unable to free sufficient memory for pipeline requiring {required_bytes/1e9:.2f}GB"
        )
        return False

    def get_pipeline[T: PipeReturn](
        self, profile: ModelProfile, expected_type: Type[T]
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

        # Build fresh pipeline with resource management
        model = self._get_model_by_id(model_id)
        if not model:
            raise RuntimeError(f"Model with ID '{model_id}' not found.")

        # Estimate memory requirements for the new pipeline
        required_memory = self._estimate_pipeline_memory_requirements(model)

        # Ensure sufficient memory is available
        if not self._ensure_sufficient_memory(required_memory, exclude_model=model_id):
            raise RuntimeError(
                f"Insufficient system resources to create pipeline for model {model.name}. "
                f"Required: {required_memory/1e9:.2f}GB"
            )

        self.logger.info(
            f"Creating pipeline for {model.name} (estimated memory: {required_memory/1e9:.2f}GB)"
        )

        pipeline = self.create_pipeline(model, profile, expected_type=expected_type)
        if not pipeline:
            raise RuntimeError(
                f"Failed to create pipeline for model {getattr(model, 'name', model_id)}."
            )

        with self._cleanup_lock:
            self._pipelines[model_id] = PipelineCacheEntry(pipeline)

        self.logger.info(f"Created and cached pipeline for model: {model.name}")

        # Update memory stats after successful creation
        hardware_manager.update_all_memory_stats()

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

            # Get memory statistics from hardware manager
            memory_stats = hardware_manager.update_all_memory_stats()

            return {
                "total_entries": len(self._pipelines),
                "alive_entries": alive_count,
                "dead_entries": len(self._pipelines) - alive_count,
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
                "gpu_count": (
                    hardware_manager.gpu_count if hardware_manager.has_gpu else 0
                ),
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
            
            # Check if model is disabled due to known issues
            if hasattr(model.details, 'disabled') and model.details.disabled:
                self.logger.info(f"Model {model.name} is disabled: {getattr(model.details, 'disable_reason', 'Unknown reason')}")
            
            # Log specific error types for better debugging
            if "unknown model architecture" in str(e):
                self.logger.error(f"Model {model.name} uses unsupported architecture - consider updating llama.cpp or using a different model")
            elif "Failed to create llama_context" in str(e):
                self.logger.error(f"Model {model.name} failed to load - may be corrupted or incompatible")
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
            from .pipelines.txt2txt.qwen3moe_v2 import QwenLangGraphPipe

            self.logger.info("Attempting to create QwenLangGraphPipe v2")
            pipeline = QwenLangGraphPipe(
                model, profile, expected_return_type=expected_type
            )
            self.logger.info("Successfully created QwenLangGraphPipe v2")
            return pipeline

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

    def force_resource_cleanup(self, target_free_memory_gb: float = 4.0) -> int:
        """
        Force cleanup of cached pipelines to free memory.

        Args:
            target_free_memory_gb: Target amount of free memory in GB

        Returns:
            Number of pipelines evicted
        """
        target_bytes = target_free_memory_gb * 1024 * 1024 * 1024

        if hardware_manager.check_memory_available(target_bytes):
            self.logger.info(f"Already have {target_free_memory_gb}GB available")
            return 0

        return len(
            [1 for _ in range(1) if self._ensure_sufficient_memory(target_bytes)]
        )

    def _cleanup_expired_entries(self) -> None:
        current_time = time.time()
        expired_keys: List[str] = []

        # Check if we're under memory pressure
        memory_pressure = False
        try:
            memory_stats = hardware_manager.update_all_memory_stats()
            for device_id, stats in memory_stats.items():
                if stats.mem_util > 85:  # More than 85% memory usage
                    memory_pressure = True
                    self.logger.info(
                        f"Memory pressure detected on GPU {device_id}: {stats.mem_util}% used"
                    )
                    break
        except Exception as e:
            self.logger.debug(f"Error checking memory pressure: {e}")

        with self._cleanup_lock:
            for model_id, entry in self._pipelines.items():
                # More aggressive cleanup under memory pressure
                cleanup_threshold = self._cache_timeout
                if memory_pressure:
                    cleanup_threshold = min(
                        self._cache_timeout, 120
                    )  # Reduce to 2 minutes under pressure

                if (
                    current_time - entry.last_accessed > cleanup_threshold
                ) or not entry.is_alive():
                    expired_keys.append(model_id)

            for model_id in expired_keys:
                entry = self._pipelines.pop(model_id, None)
                if entry:
                    pipeline = entry.pipeline
                    if pipeline:
                        self._cleanup_pipeline_resources(pipeline)
                    pressure_note = " (memory pressure)" if memory_pressure else ""
                    self.logger.info(
                        f"Removed expired pipeline for model {model_id}{pressure_note}"
                    )

        # If under memory pressure, also do hardware cleanup
        if memory_pressure and expired_keys:
            hardware_manager.clear_memory(aggressive=True)

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
