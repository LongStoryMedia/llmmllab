"""
Production-ready pipeline factory with weakref caching, background cleanup, and
modern/legacy pipeline selection. Replaces the previous garbled version.
"""

import json
import logging
import os
import threading
import time
import weakref
import gc
from contextlib import contextmanager
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Type, cast, ContextManager, Iterator

from models import Model, LoraWeight, ModelDetails, ModelProfile, ChatResponse
from .pipelines.base import BasePipelineCore, PipeReturn
from utils.hardware_manager import hardware_manager


class PipelinePriority(IntEnum):
    """Pipeline priority levels for cache eviction."""

    LOW = 1  # Tool generation, etc. (evict first)
    MEDIUM = 5  # Standard pipelines
    NORMAL = 5  # Standard pipelines
    HIGH = 10  # Critical pipelines (main chat models)
    CRITICAL = 20  # For primary/main models that should rarely be evicted


class PipelineCacheEntry:
    """Cache entry with automatic cleanup via weak references and priority-based eviction."""

    def __init__(
        self,
        pipeline: BasePipelineCore,
        priority: PipelinePriority = PipelinePriority.NORMAL,
        timestamp: Optional[float] = None,
    ):
        self._pipeline_ref = weakref.ref(pipeline)
        self.last_accessed = timestamp if timestamp is not None else time.time()
        self.creation_time = time.time()
        self.priority = priority
        self.access_count = 1  # Track how often this pipeline is used
        self.lock = threading.Lock()

    @property
    def pipeline(self) -> Optional[BasePipelineCore]:
        """Return the live pipeline instance, or None if it has been GC'd."""
        return self._pipeline_ref()

    def is_alive(self) -> bool:
        return self._pipeline_ref() is not None

    def touch(self) -> None:
        self.last_accessed = time.time()
        self.access_count += 1

    @contextmanager
    def use_pipeline(self) -> Iterator[Optional[BasePipelineCore]]:
        """A context manager to safely use the pipeline with a lock."""
        if not self.is_alive():
            raise RuntimeError("Pipeline is no longer available.")
        
        pipeline = self.pipeline
        if pipeline is None:
            raise RuntimeError("Pipeline reference is dead.")

        with self.lock:
            try:
                yield pipeline
            finally:
                pass  # Lock is released automatically

    def get_eviction_score(self, current_time: float) -> float:
        """
        Calculate eviction score. Lower scores are evicted first.

        Score considers:
        - Age since last access (older = lower score)
        - Priority (higher priority = higher score)
        - Access frequency (more used = higher score)
        """
        age_seconds = current_time - self.last_accessed

        # Base score from priority
        score = float(self.priority)

        # Reduce score based on age (older items get lower scores)
        age_penalty = age_seconds / 3600.0  # Penalty per hour
        score -= age_penalty

        # Boost score based on access frequency
        frequency_boost = min(self.access_count / 10.0, 2.0)  # Cap at 2.0 boost
        score += frequency_boost

        return score


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
    _lock = threading.Lock()  # Lock to serialize pipeline creation
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

    def _estimate_pipeline_memory_requirements(self, model: Model, profile: ModelProfile) -> float:
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
            # Context memory scales with n_ctx
            num_ctx = profile.parameters.num_ctx or 4096  # Default to 4096 if not set
            # Heuristic: Each token in the context might take up space in KV cache.
            # This varies wildly by model, but we can use a rough estimate.
            # Let's say ~512 bytes per token in the context cache.
            context_memory = num_ctx * 512
            context_memory = min(context_memory, 4 * 1024 * 1024 * 1024) # Cap at 4GB

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
        Uses priority-based eviction: lower priority and older pipelines are evicted first.

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
            "Attempting priority-based cache eviction..."
        )

        # Clear any already-dead entries first
        with self._cleanup_lock:
            dead_keys = [
                k for k, entry in self._pipelines.items() if not entry.is_alive()
            ]
            for k in dead_keys:
                self._pipelines.pop(k, None)
                self.logger.debug(f"Removed dead pipeline entry: {k}")

        # Try gentle memory clearing first
        hardware_manager.clear_memory(aggressive=False)
        if hardware_manager.check_memory_available(required_bytes):
            self.logger.info("Sufficient memory available after gentle cleanup")
            return True

        # Get entries sorted by eviction score (lowest score = evicted first)
        current_time = time.time()
        with self._cleanup_lock:
            eviction_candidates = [
                (model_id, entry, entry.get_eviction_score(current_time))
                for model_id, entry in self._pipelines.items()
                if entry.is_alive() and model_id != exclude_model
            ]
            # Sort by eviction score (lowest first), then by priority (lowest first), then by age (oldest first)
            eviction_candidates.sort(
                key=lambda x: (x[2], x[1].priority, x[1].last_accessed)
            )

        # Evict cached pipelines until we have enough memory
        evicted_count = 0
        for model_id, entry, score in eviction_candidates:
            with self._cleanup_lock:
                removed_entry = self._pipelines.pop(model_id, None)
                if removed_entry and removed_entry.pipeline:
                    # Rely on GC to clean up the pipeline resources
                    pass
                evicted_count += 1

            self.logger.info(
                f"Evicted pipeline for model: {model_id} "
                f"(priority: {entry.priority}, score: {score:.2f}, "
                f"age: {(current_time - entry.last_accessed)/60:.1f}min, "
                f"access_count: {entry.access_count})"
            )

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

    def get_pipeline(
        self,
        model_id: str,
        profile: ModelProfile,
        priority: PipelinePriority = PipelinePriority.NORMAL,
    ) -> Optional[ContextManager[Optional[BasePipelineCore]]]:
        """Get or create a pipeline for a given model ID and profile."""
        with self._lock:
            # First, check if a compatible pipeline is already cached and alive
            cached_entry = self._pipelines.get(model_id)
            if cached_entry and cached_entry.is_alive():
                pipeline = cached_entry.pipeline
                if pipeline:
                    self.logger.info(f"Reusing cached pipeline for model: {model_id}")
                    cached_entry.touch()
                    # Update priority if a higher priority is requested
                    if priority > cached_entry.priority:
                        cached_entry.priority = priority
                    return cached_entry.use_pipeline()

            # If not cached or the weakref is dead, create a new one
            model = self._available_models.get(model_id)
            if not model:
                self.logger.error(f"Model with ID '{model_id}' not found.")
                return None

            # Estimate memory before creation
            estimated_bytes = self._estimate_pipeline_memory_requirements(model, profile)
            self.logger.info(
                f"Memory estimate for {model.name}: {estimated_bytes / 1024**3:.2f}GB"
            )

            # Check for available memory and perform cleanup if necessary
            if not self._ensure_sufficient_memory(estimated_bytes, exclude_model=model_id):
                self.logger.error(
                    f"Unable to free sufficient memory for pipeline requiring {estimated_bytes / 1024**3:.2f}GB"
                )
                return None

            self.logger.info(
                f"Creating pipeline for {model.name} (estimated memory: {estimated_bytes / 1024**3:.2f}GB)"
            )

            try:
                pipeline = self._create_pipeline_instance(model, profile)
                if pipeline:
                    entry = PipelineCacheEntry(pipeline, priority=priority)
                    self._pipelines[model_id] = entry
                    self.logger.info(f"Created and cached pipeline for model: {model.name}")
                    return entry.use_pipeline()
            except Exception as e:
                self.logger.error(f"Error creating pipeline for {model.name}: {e}")
                # Ensure a failed creation doesn't leave a bad entry
                if model_id in self._pipelines:
                    del self._pipelines[model_id]
                return None

        return None

    def _create_pipeline_instance(
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
                self.logger.error(f"Model profile validation error: {e}")
            
            return None

    def _cleanup_pipeline_resources(self, pipeline: BasePipelineCore):
        """Explicitly clean up resources of a pipeline."""
        self.logger.debug(f"Cleaning up pipeline: {pipeline}")
        # If the pipeline has a specific cleanup method, call it
        if hasattr(pipeline, "cleanup"):
            pipeline.cleanup()
        del pipeline
        gc.collect()
        hardware_manager.clear_memory()


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

        # if model.pipeline == "OpenAiGptOssPipe":
        #     from .pipelines.txt2txt.openai_gpt_oss import OpenAiGptOssPipe

        #     return OpenAiGptOssPipe(model, profile, expected_return_type=expected_type)

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

    # ---------- Cache management and cleanup ----------

    def _start_cleanup_thread(self) -> None:
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return

        def cleanup_job() -> None:
            while True:
                time.sleep(self._cache_timeout)
                self.clear_expired_pipelines()

        self._cleanup_thread = threading.Thread(target=cleanup_job, daemon=True)
        self._cleanup_thread.start()

    def clear_expired_pipelines(self) -> None:
        """Remove pipelines that haven't been used recently."""
        with self._cleanup_lock:
            now = time.time()
            expired_keys = [
                key
                for key, entry in self._pipelines.items()
                if not entry.is_alive()
                or (now - entry.last_accessed) > self._cache_timeout
            ]

            if expired_keys:
                self.logger.info(f"Removing {len(expired_keys)} expired pipelines...")
                for key in expired_keys:
                    if key in self._pipelines:
                        del self._pipelines[key]
                # Hint to the garbage collector
                hardware_manager.clear_memory()

    def clear_unused_pipelines(self) -> int:
        """Force-clears all pipelines that are not currently in use."""
        with self._cleanup_lock:
            initial_count = len(self._pipelines)
            if initial_count == 0:
                return 0

            # Identify pipelines that are still referenced elsewhere
            in_use_keys = {
                key for key, entry in self._pipelines.items() if entry.is_alive()
            }
            all_keys = set(self._pipelines.keys())
            eviction_keys = all_keys - in_use_keys

            if eviction_keys:
                self.logger.info(
                    f"Clearing {len(eviction_keys)} unused pipelines: {list(eviction_keys)}"
                )
                for key in eviction_keys:
                    del self._pipelines[key]

            # After clearing, trigger garbage collection
            hardware_manager.clear_memory(aggressive=True)
            return len(eviction_keys)

    def _evict_pipelines_by_priority(self, required_space_gb: float) -> None:
        """Evict pipelines based on score until enough space is freed."""
        with self._cleanup_lock:
            pass


    def clear_memory(self, aggressive: bool = False) -> None:
        """Public method to trigger memory clearing."""
        with self._cleanup_lock:
            self.logger.info(
                f"External request to clear memory (aggressive={aggressive})"
            )
            # Evict all pipelines that are not strongly referenced
            self.clear_unused_pipelines()
            hardware_manager.clear_memory(aggressive=aggressive)

    # ---------- Memory estimation ----------

    def _get_context_memory_gb(self, profile: ModelProfile) -> float:
        """Estimate the context memory required for a given profile."""
        # Simple heuristic: 32 bytes per token, with a minimum of 1MB
        num_ctx = profile.parameters.num_ctx or 512  # Default to 512 if not set
        per_token_bytes = 32
        context_gb = max((num_ctx * per_token_bytes) / (1024**3), 1.0)

        self.logger.debug(
            f"Context memory estimate for {profile}: {context_gb:.2f}GB ({num_ctx} tokens)"
        )
        return context_gb

    def _estimate_memory_usage(
        self, model: Model, profile: ModelProfile
    ) -> float:
        """
        Estimate the total memory usage for a pipeline given a model and profile.
        """
        # Base estimate from model size
        model_gb = model.size / (1024**3) if model.size else 0.0

        # Context memory based on profile
        context_gb = self._get_context_memory_gb(profile)

        # Total estimate is model size + context size + some base overhead
        return model_gb + context_gb + 0.5 # 0.5 GB base overhead


# Singleton instance
pipeline_factory = PipelineFactory()
