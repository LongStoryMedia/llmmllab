"""
Local pipeline cache & memory management for local model providers.

Extracted from pipeline_factory so only local (on-device) model providers
consume persistent cached resources. Remote/API providers bypass caching.
"""

from __future__ import annotations

import threading
import time
import weakref
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Type, Generator

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

from runner.models import (
    Model,
    ModelProfile,
    ModelProvider,
    PipelinePriority,
    UserConfig,
    OptimalParameters,
    ModelParameters,
)
from runner.pipelines.base import BasePipeline
from runner.utils.logging import llmmllogger
from .utils.hardware_manager import hardware_manager
from .utils.resizer import Resizer
from .utils.intelligent_oom_recovery import IntelligentOOMRecovery


class _PipelineCacheEntry:

    def __init__(
        self,
        pipeline: BasePipeline | Embeddings,
        priority: PipelinePriority,
        estimated_memory: float = 0,
    ):
        self._ref = weakref.ref(pipeline)
        # Strong reference keeps pipeline alive in cache.
        # Cleared only on explicit eviction or cache removal.
        self._strong_ref: Optional[BasePipeline | Embeddings] = pipeline
        self.priority = priority
        self.estimated_memory = (
            estimated_memory  # Store memory estimate for eviction decisions
        )
        self.creation_time = time.time()
        self.last_accessed = self.creation_time
        self.access_count = 1
        self.in_use = False  # Prevent eviction while pipeline is actively generating
        self._use_count = 0  # Track concurrent usage
        self.persistent = False  # Never expire via timeout (OOM eviction still applies)

    @property
    def pipeline(self) -> Optional[BasePipeline | Embeddings]:
        # Prefer strong ref when locked, fall back to weakref
        if self._strong_ref is not None:
            return self._strong_ref
        return self._ref()

    @property
    def use_count(self) -> int:
        """Get the current use count for this pipeline."""
        return self._use_count

    def is_alive(self) -> bool:
        if self._strong_ref is not None:
            return True
        return self._ref() is not None

    def touch(self) -> None:
        self.last_accessed = time.time()
        self.access_count += 1

    def lock(self) -> None:
        """Mark pipeline as in-use to prevent eviction."""
        self._use_count += 1
        self.in_use = True
        self.touch()

    def unlock(self) -> None:
        """Release pipeline from in-use state."""
        self._use_count = max(0, self._use_count - 1)
        self.in_use = self._use_count > 0

    def eviction_score(self, now: float, estimated_memory: float = 0) -> float:
        """Calculate eviction score - higher score = keep longer, lower score = evict first."""
        # Age penalty (older = more likely to evict)
        age_penalty = (now - self.last_accessed) / 3600.0

        # Priority bonus (higher priority = keep longer)
        priority_bonus = float(self.priority.value) * 2.0

        # Access frequency bonus (more used = keep longer)
        access_bonus = min(self.access_count / 5.0, 3.0)

        # Memory efficiency bonus (smaller models get bonus to stay)
        # Small models (< 2GB) get significant bonus, large models (> 10GB) get penalty
        if estimated_memory > 0:
            if estimated_memory < 2 * 1024**3:  # < 2GB (embeddings, small models)
                memory_bonus = 5.0  # Strong preference to keep small models
                # Extra tiny bonus for very small models to break ties
                if estimated_memory < 1 * 1024**3:  # < 1GB
                    memory_bonus += 0.1
            elif estimated_memory < 5 * 1024**3:  # < 5GB (medium models)
                memory_bonus = 2.0
            elif estimated_memory < 10 * 1024**3:  # < 10GB (large models)
                memory_bonus = 0.0
            else:  # >= 10GB (very large models)
                memory_bonus = -2.0  # Slight penalty for very large models
        else:
            memory_bonus = 0.0

        score = priority_bonus + access_bonus + memory_bonus - age_penalty
        return score


class LocalPipelineCacheManager:
    """Caches pipelines only for local providers (llama.cpp, stable diffusion cpp)."""

    LOCAL_PROVIDERS = {ModelProvider.LLAMA_CPP, ModelProvider.STABLE_DIFFUSION_CPP}

    def __init__(self, cache_timeout: int = 300):
        self._cache: Dict[str, _PipelineCacheEntry] = {}
        self._lock = threading.RLock()
        self._cache_timeout = cache_timeout
        self.logger = llmmllogger.logger.bind(component="LocalPipelineCacheManager")
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Initialize the resizer and OOM recovery components
        self._resizer = Resizer()
        try:
            # Use a local directory for development, /app for production
            import os

            if os.path.exists("/app"):
                self._oom_recovery = IntelligentOOMRecovery()
            else:
                # Development environment - use local directory
                import tempfile

                local_data_dir = os.path.join(
                    tempfile.gettempdir(), "oom_recovery_data"
                )
                self._oom_recovery = IntelligentOOMRecovery(data_dir=local_data_dir)
        except Exception as e:
            self.logger.warning(
                f"Failed to initialize OOM recovery: {e}, disabling graceful degradation"
            )
            self._oom_recovery = None

        self._start_cleanup_thread()

    # ---- Public API ----
    def is_local(self, model: Model) -> bool:
        try:
            return model.provider in self.LOCAL_PROVIDERS  # type: ignore[attr-defined]
        except Exception:
            return False

    def get_or_create(
        self,
        model: Model,
        profile: ModelProfile,
        priority: PipelinePriority,
        create_fn: Callable[
            [Model, ModelProfile, Optional[Type[BaseModel]], Optional[dict]],
            Optional[BasePipeline | Embeddings],
        ],
        grammar: Optional[Type[BaseModel]] = None,
        user_config: Optional[UserConfig] = None,
        metadata: Optional[dict] = None,
    ) -> BasePipeline | Embeddings:
        assert profile.id is not None, "ModelProfile must have a valid ID"
        profile_id = str(profile.id)

        with self._lock:
            entry = self._cache.get(profile_id)
            if entry and entry.is_alive():
                pipe = entry.pipeline
                if pipe:
                    is_healthy = True
                    if hasattr(pipe, "server_manager"):
                        # Check server status if available
                        sm = getattr(pipe, "server_manager", None)
                        if sm and hasattr(sm, "is_running") and not sm.is_running():
                            self.logger.warning(
                                f"⚠️ Found cached pipeline for {profile_id} with dead server - evicting"
                            )
                            is_healthy = False

                    if is_healthy:
                        entry.touch()
                        if isinstance(pipe, BasePipeline) and metadata:
                            pipe.bind_metadata(metadata)
                        self.logger.debug(
                            f"💾 Retrieved cached pipeline for {profile_id}"
                        )
                        return pipe

                # If we reach here, the entry exists but is either:
                # 1. Dead (weakref is None)
                # 2. Unhealthy (server is dead)
                # So we remove it.
                self._cache.pop(profile_id, None)

        self.logger.info(f"🆕 Creating new pipeline for {profile_id}")
        # Estimate memory requirement for this model/profile
        required = self.estimate_memory(model, profile)

        # Check if graceful degradation is enabled and try OOM recovery if needed
        if not self._ensure_memory(required, exclude=profile_id):
            # Check if we should try intelligent OOM recovery
            if (
                user_config
                and user_config.parameter_optimization
                and user_config.parameter_optimization.crash_prevention
                and user_config.parameter_optimization.crash_prevention.enable_graceful_degradation
                and self._oom_recovery is not None
            ):

                self.logger.info(
                    f"🔄 Insufficient memory for {model.name} ({required/1e9:.2f}GB), "
                    f"attempting graceful degradation via OOM recovery"
                )

                try:
                    # Use OOM recovery to get optimized parameters for the current hardware
                    optimized_params = (
                        self._oom_recovery.predict_optimal_parameters_from_profile(
                            model, profile
                        )
                    )

                    for param, value in optimized_params.model_dump().items():
                        setattr(profile.parameters, param, value)

                    # Re-estimate memory with optimized parameters
                    optimized_required = self.estimate_memory(model, profile)

                    self.logger.info(
                        f"🎯 OOM recovery optimized memory requirement from {required/1e9:.2f}GB to {optimized_required/1e9:.2f}GB"
                    )

                    # Try again with optimized parameters
                    if self._ensure_memory(optimized_required, exclude=profile_id):
                        required = optimized_required
                        self.logger.info(
                            "✅ OOM recovery successful, proceeding with optimized parameters"
                        )
                    else:
                        self.logger.error(
                            "❌ OOM recovery failed - still insufficient memory even after optimization"
                        )
                        raise RuntimeError(
                            f"Insufficient memory for local model {model.name}: need {optimized_required/1e9:.2f}GB even after optimization"
                        )

                except Exception as e:
                    self.logger.error(f"❌ OOM recovery failed with error: {e}")
                    raise RuntimeError(
                        f"Insufficient memory for local model {model.name}: need {required/1e9:.2f}GB, OOM recovery failed: {e}"
                    ) from e
            else:
                # No graceful degradation, raise error immediately
                raise RuntimeError(
                    f"Insufficient memory for local model {model.name}: need {required/1e9:.2f}GB"
                )

        pipeline = create_fn(model, profile, grammar, metadata)
        if not pipeline:
            raise RuntimeError(f"Failed to create pipeline for {model.name}")

        with self._lock:
            self._cache[profile_id] = _PipelineCacheEntry(pipeline, priority, required)
            self.logger.debug(f"💾 Cached NEW pipeline for {profile_id}")

        # Auto-mark small models (likely embeddings) as persistent — they're tiny
        # and cheap to keep loaded indefinitely.
        if required < 2 * 1024 * 1024 * 1024:  # < 2GB
            self.set_persistent(profile_id, True)
            self.logger.info(f"🔒 Auto-marked small model {profile_id} as persistent")

        hardware_manager.update_all_memory_stats()
        return pipeline

    def clear_cache(self, model_id: Optional[str] = None) -> None:
        with self._lock:
            targets = [model_id] if model_id else list(self._cache.keys())
            for mid in targets:
                entry = self._cache.pop(mid, None)
                if entry and entry.pipeline:
                    self._cleanup_pipeline(entry.pipeline)
        self.logger.info(
            "Cleared %s local pipeline cache entries",
            "all" if model_id is None else model_id,
        )

    def clear_expired(self) -> None:
        """Clear expired entries using intelligent timeout based on pipeline characteristics."""
        now = time.time()
        expired: List[str] = []
        with self._lock:
            for mid, entry in self._cache.items():
                if not entry.is_alive():
                    expired.append(mid)
                    continue

                # Never time-out persistent entries; OOM eviction handles them if needed
                if entry.persistent:
                    continue

                # Calculate dynamic timeout based on pipeline characteristics
                base_timeout = self._cache_timeout

                # Small models get much longer timeout (they're cheap to keep)
                if entry.estimated_memory < 2 * 1024 * 1024 * 1024:  # < 2GB
                    timeout_multiplier = 10.0  # 10x longer timeout for small models
                elif entry.estimated_memory < 5 * 1024 * 1024 * 1024:  # < 5GB
                    timeout_multiplier = 3.0  # 3x longer for medium models
                else:
                    # Large inference models (>= 5GB): 6x base = 30 min inactivity timeout.
                    # Long enough to survive pauses in a coding session; short enough to
                    # unload when the user is genuinely done and free the GPU.
                    timeout_multiplier = 6.0

                # High priority models get longer timeout
                if entry.priority.value >= 4:  # HIGH or URGENT priority
                    timeout_multiplier *= 2.0

                # Frequently accessed models get longer timeout
                if entry.access_count > 5:
                    timeout_multiplier *= 1.5

                dynamic_timeout = base_timeout * timeout_multiplier

                if (now - entry.last_accessed) > dynamic_timeout:
                    # Never expire a locked (in-use) pipeline
                    if entry.in_use:
                        self.logger.debug(
                            f"Skipping expiry of {mid} - pipeline is in use "
                            f"(use_count: {entry.use_count})"
                        )
                        continue
                    self.logger.debug(
                        f"Expiring {mid} after {dynamic_timeout:.0f}s timeout "
                        f"(base: {base_timeout}s, multiplier: {timeout_multiplier:.1f}x, "
                        f"mem: {entry.estimated_memory/1e9:.2f}GB, priority: {entry.priority.name})"
                    )
                    expired.append(mid)

            for mid in expired:
                removed = self._cache.pop(mid, None)
                if removed and removed.pipeline:
                    self._cleanup_pipeline(removed.pipeline)
        if expired:
            self.logger.debug(f"Expired local pipelines cleared: {expired}")

    def stats(self) -> Dict[str, Any]:  # noqa: ANN401
        with self._lock:
            alive = {mid: e for mid, e in self._cache.items() if e.is_alive()}
            locked_count = sum(1 for e in alive.values() if e.in_use)
            mem = hardware_manager.update_all_memory_stats()
            return {
                "count": len(self._cache),
                "alive": len(alive),
                "dead": len(self._cache) - len(alive),
                "locked": locked_count,
                "entries": {
                    mid: {
                        "priority": e.priority.name,
                        "access_count": e.access_count,
                        "last_accessed": e.last_accessed,
                        "in_use": e.in_use,
                        "use_count": e.use_count,
                        "estimated_memory_gb": (
                            e.estimated_memory / 1e9 if e.estimated_memory else 0
                        ),
                    }
                    for mid, e in alive.items()
                },
                "memory": {
                    dev: {
                        "total_mb": s.mem_total,
                        "used_mb": s.mem_used,
                        "free_mb": s.mem_free,
                        "util_percent": s.mem_util,
                    }
                    for dev, s in mem.items()
                },
            }

    def set_priority(self, model_id: str, priority: PipelinePriority) -> bool:
        with self._lock:
            entry = self._cache.get(model_id)
            if entry and entry.is_alive():
                entry.priority = priority
                return True
        return False

    def lock_pipeline(self, model_id: str) -> bool:
        """Lock a pipeline to prevent eviction during active use."""
        with self._lock:
            entry = self._cache.get(model_id)
            if entry and entry.is_alive():
                entry.lock()
                self.logger.debug(f"🔒 Locked pipeline {model_id} for active use")
                return True
        return False

    def unlock_pipeline(self, model_id: str) -> bool:
        """Unlock a pipeline when no longer actively in use."""
        with self._lock:
            entry = self._cache.get(model_id)
            if entry and entry.is_alive():
                entry.unlock()
                self.logger.debug(f"🔓 Unlocked pipeline {model_id}")
                return True
        return False

    def set_persistent(self, model_id: str, persistent: bool = True) -> bool:
        """Mark a pipeline as persistent (should avoid eviction unless absolutely necessary)."""
        with self._lock:
            entry = self._cache.get(model_id)
            if entry and entry.is_alive():
                entry.persistent = persistent
                if persistent:
                    entry.touch()
                    self.logger.info(f"🔒 Marked pipeline {model_id} as persistent")
                else:
                    self.logger.info(
                        f"🔓 Removed persistent marking from pipeline {model_id}"
                    )
                return True
        return False

    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information for monitoring and debugging."""
        with self._lock:
            alive = {mid: e for mid, e in self._cache.items() if e.is_alive()}
            total_memory = sum(e.estimated_memory for e in alive.values())
            small_models = {
                mid: e for mid, e in alive.items() if e.estimated_memory < 2 * 1024**3
            }
            large_models = {
                mid: e for mid, e in alive.items() if e.estimated_memory >= 10 * 1024**3
            }

            return {
                "total_models": len(alive),
                "total_memory_gb": total_memory / 1e9,
                "small_models": {
                    "count": len(small_models),
                    "memory_gb": sum(e.estimated_memory for e in small_models.values())
                    / 1e9,
                    "models": list(small_models.keys()),
                },
                "large_models": {
                    "count": len(large_models),
                    "memory_gb": sum(e.estimated_memory for e in large_models.values())
                    / 1e9,
                    "models": list(large_models.keys()),
                },
                "locked_models": [mid for mid, e in alive.items() if e.in_use],
                "high_priority_models": [
                    mid for mid, e in alive.items() if e.priority.value >= 4
                ],
            }

    @contextmanager
    def pipeline_in_use(self, model_id: str) -> Generator[bool, None, None]:
        """
        Context manager to safely lock/unlock a pipeline during active use.

        Usage:
            with cache_manager.pipeline_in_use(model_id) as locked:
                if locked:
                    # Pipeline is locked, safe to use
                    result = await pipeline.generate(...)
                # Pipeline automatically unlocked when exiting context
        """
        locked = self.lock_pipeline(model_id)
        try:
            yield locked
        finally:
            if locked:
                self.unlock_pipeline(model_id)

    def force_cleanup(self) -> int:
        with self._lock:
            count = len(self._cache)
            for mid, entry in list(self._cache.items()):
                self._cache.pop(mid, None)
                if entry and entry.pipeline:
                    self._cleanup_pipeline(entry.pipeline)
        hardware_manager.clear_memory()
        return count

    def cleanup_for_user(self, user_config: UserConfig) -> int:
        """Cleanup all pipelines associated with a specific user."""
        cleaned_count = 0
        pids_to_cleanup = []

        with self._lock:
            # First pass: collect all PIDs and clean up pipelines
            # FIX: Iterate over ALL model fields, not just set ones
            # This ensures default profile IDs (like Chat/Primary) are caught even if not explicitly set in DB
            for model_profile_field in user_config.model_profiles.model_fields:
                profile_id = getattr(user_config.model_profiles, model_profile_field)
                if not profile_id:
                    continue

                profile_id = str(profile_id)
                entry = self._cache.get(profile_id)

                if entry and entry.pipeline:
                    # Unlock the pipeline before cleanup
                    if entry.in_use:
                        entry.unlock()
                        self.logger.info(
                            f"Unlocked pipeline {profile_id} before cleanup"
                        )

                    # Extract PID if it's a server-based pipeline
                    pid = None
                    if isinstance(entry.pipeline, BasePipeline):
                        if hasattr(entry.pipeline, "server_manager") and entry.pipeline.server_manager:  # type: ignore[attr-defined]
                            pid = entry.pipeline.server_manager.pid  # type: ignore[attr-defined]
                            if pid:
                                pids_to_cleanup.append(pid)
                                self.logger.info(
                                    f"Collected PID {pid} for cleanup from pipeline {profile_id}"
                                )

                    # Now remove from cache
                    self._cache.pop(profile_id, None)

                    # Shutdown the pipeline (stops the server gracefully)
                    self._cleanup_pipeline(entry.pipeline)
                    cleaned_count += 1

            # Second pass: Kill any remaining processes and clear GPU memory
            if pids_to_cleanup:
                self.logger.info(
                    f"Performing GPU cleanup for {len(pids_to_cleanup)} process(es): {pids_to_cleanup}"
                )

                # Kill each PID individually on each GPU it's using
                for pid in pids_to_cleanup:
                    for device_idx in range(
                        hardware_manager.gpu_count if hardware_manager.has_gpu else 0
                    ):
                        # Check if this PID is on this GPU
                        processes = hardware_manager.get_gpu_process_info(
                            device_idx
                        ).get(device_idx, [])
                        if any(p["pid"] == pid for p in processes):
                            self.logger.info(f"Clearing GPU {device_idx} for PID {pid}")
                            # Nuclear clear for this specific PID only, with reinitialization
                            hardware_manager.clear_memory(
                                device_idx=device_idx, pid=pid
                            )

            self.logger.info("Completed GPU cleanup for all collected PIDs")

        return cleaned_count

    # ---- Internals ----
    def _convert_model_parameters_to_optimal(
        self, model_params: ModelParameters
    ) -> OptimalParameters:
        """Convert ModelParameters to OptimalParameters for use with Resizer."""
        # Apply reasonable defaults and log warnings for extreme values
        context_size = model_params.num_ctx or 4096
        batch_size = model_params.batch_size or 512

        # Log warning for extremely large context sizes that might cause issues
        if context_size > 65536:  # 64K tokens
            self.logger.warning(
                f"⚠️ Very large context size ({context_size:,}) may cause high memory usage"
            )

        return OptimalParameters(
            n_ctx=context_size,
            n_batch=batch_size,
            n_ubatch=128,  # Default micro-batch size (not in ModelParameters)
            n_gpu_layers=-1,  # Default to all layers on GPU (not in ModelParameters)
        )

    def estimate_memory(
        self, model: Model, profile: Optional["ModelProfile"] = None
    ) -> float:
        """Estimate memory usage using corrected formulas that match real-world llama.cpp usage."""
        if not profile or not profile.parameters:
            # Fallback to simple estimation if no profile provided
            base = 512 * 1024 * 1024  # 512MB base
            model_size = getattr(model, "size", 0)
            if model_size == 0:
                # Fallback estimation based on task
                task = str(getattr(model, "task", "TextToText"))
                if task.endswith("TextToEmbeddings"):
                    model_size = 1 * 1024 * 1024 * 1024  # 1GB
                else:
                    model_size = 4 * 1024 * 1024 * 1024  # 4GB

            total = base + model_size + (model_size * 0.2)  # 20% context overhead
            self.logger.debug(
                f"Basic memory estimate for {model.name}: {total/1e9:.2f}GB "
                f"(no profile available, using fallback)"
            )
            return total

        try:
            # Use corrected memory estimation based on real-world data
            memory_breakdown = self._calculate_corrected_memory_breakdown(
                profile.parameters, model
            )

            # Total GPU memory estimate
            total_memory = (
                memory_breakdown["total_gb"] * 1024 * 1024 * 1024
            )  # Convert to bytes

            self.logger.debug(
                f"Corrected memory estimate for {model.name}: {total_memory/1e9:.2f}GB "
                f"(model: {memory_breakdown['model_weights_gb']:.2f}GB, "
                f"kv_cache: {memory_breakdown['kv_cache_gb']:.2f}GB, "
                f"activation: {memory_breakdown['activation_gb']:.2f}GB, "
                f"overhead: {memory_breakdown['overhead_gb']:.2f}GB)"
            )

            return total_memory

        except Exception as e:
            self.logger.warning(
                f"Failed to use corrected memory estimation: {e}, falling back to basic estimation"
            )
            # Fallback to basic estimation
            model_size = getattr(model, "size", 4 * 1024 * 1024 * 1024)
            context_mem = model_size * 0.3  # 30% for context
            total = model_size + context_mem + (512 * 1024 * 1024)  # Add 512MB overhead

            self.logger.debug(
                f"Fallback memory estimate for {model.name}: {total/1e9:.2f}GB"
            )
            return total

    def _calculate_corrected_memory_breakdown(
        self, params: ModelParameters, model: Model
    ) -> dict:
        """Calculate memory breakdown using Resizer."""
        optimal_params = self._convert_model_parameters_to_optimal(params)
        breakdown = self._resizer.calculate_memory_breakdown(optimal_params, model)
        return {
            "model_weights_gb": breakdown["model_weights_gpu_gb"],
            "kv_cache_gb": breakdown["kv_cache_gb"],
            "activation_gb": breakdown["activation_gb"],
            "overhead_gb": breakdown["overhead_gb"],
            "clip_model_gb": breakdown["clip_model_gb"],
            "total_gb": breakdown["total_gpu_gb"],
            "kv_efficiency": 1.0,  # Not used in breakdown
            "gpu_layers": breakdown["gpu_layers_loaded"],
        }

    def _ensure_memory(self, required: float, exclude: Optional[str]) -> bool:
        """Ensure sufficient memory is available, with intelligent eviction based on size and priority."""

        # Check if we already have enough memory - avoid unnecessary eviction
        if hardware_manager.check_memory_available(required):
            self.logger.debug(
                f"✅ Sufficient memory available ({required/1e9:.2f}GB), no eviction needed"
            )
            return True

        self.logger.info(f"🔍 Need {required/1e9:.2f}GB, checking eviction candidates")

        # For very large models (>15GB), be more aggressive about clearing space
        large_model = required > 15 * 1024 * 1024 * 1024  # 15GB threshold
        if large_model:
            self.logger.info(
                f"🚀 Very large model detected ({required/1e9:.2f}GB), using aggressive eviction"
            )
            # Clear most models immediately for very large models, except small ones and those in use
            with self._lock:
                evict_targets = []
                keep_targets = []
                locked_targets = []

                for mid, entry in self._cache.items():
                    if mid == exclude:
                        continue
                    if entry.in_use:
                        locked_targets.append(mid)
                        continue
                    # Keep small models (< 3GB) even for large model loads
                    if entry.estimated_memory < 3 * 1024 * 1024 * 1024:  # 3GB threshold
                        keep_targets.append((mid, entry.estimated_memory / 1e9))
                        continue
                    evict_targets.append(mid)

            if keep_targets:
                self.logger.info(
                    f"🛡️ Keeping {len(keep_targets)} small models: {[(mid, f'{mem:.2f}GB') for mid, mem in keep_targets]}"
                )
            if locked_targets:
                self.logger.warning(
                    f"⚠️ Cannot evict {len(locked_targets)} models currently in use: {locked_targets}"
                )

            if evict_targets:
                self.logger.info(
                    f"🧹 Aggressively evicting {len(evict_targets)} large models: {evict_targets}"
                )
                for mid in evict_targets:
                    with self._lock:
                        removed = self._cache.pop(mid, None)
                    if removed and removed.pipeline:
                        self._cleanup_pipeline(removed.pipeline)

                # Aggressive memory clear after eviction
                hardware_manager.clear_memory()
                self.logger.info(
                    "🧹 Completed aggressive cache clearing for very large model"
                )

        # Check if we now have enough memory
        if hardware_manager.check_memory_available(required):
            return True

        self.logger.info(
            f"💡 Memory still needed after initial clearing, using intelligent eviction (need {required/1e9:.2f}GB)"
        )

        # Step 1: Clear dead entries first
        with self._lock:
            dead = [mid for mid, e in self._cache.items() if not e.is_alive()]
            for mid in dead:
                self._cache.pop(mid, None)

        if dead:
            self.logger.debug(f"🗑️ Cleared {len(dead)} dead entries")
            hardware_manager.clear_memory()
            if hardware_manager.check_memory_available(required):
                return True

        # Step 2: Intelligent eviction by enhanced scoring
        now = time.time()
        with self._lock:
            candidates = []
            protected = []
            locked_pipelines = []

            for mid, entry in self._cache.items():
                if not entry.is_alive() or mid == exclude:
                    continue

                if entry.in_use:
                    locked_pipelines.append((mid, entry.estimated_memory / 1e9))
                    continue

                eviction_score = entry.eviction_score(now, entry.estimated_memory)

                # Protect small, high-value pipelines from eviction unless absolutely necessary
                if (
                    entry.estimated_memory < 1.5 * 1024 * 1024 * 1024  # < 1.5GB
                    and entry.priority.value >= 3  # Medium priority or higher
                    and entry.access_count > 2
                ):  # Used multiple times
                    protected.append(
                        (mid, entry.estimated_memory / 1e9, eviction_score)
                    )
                    continue

                candidates.append(
                    (mid, entry, eviction_score, entry.estimated_memory / 1e9)
                )

        if protected:
            self.logger.info(
                f"🛡️ Protected {len(protected)} small/valuable models from eviction: {[(mid, f'{mem:.2f}GB', f'score:{score:.1f}') for mid, mem, score in protected]}"
            )
        if locked_pipelines:
            self.logger.warning(
                f"⚠️ Skipping {len(locked_pipelines)} locked pipelines during eviction: {[(mid, f'{mem:.2f}GB') for mid, mem in locked_pipelines]}"
            )

        # Sort candidates by eviction score (lowest score = evict first)
        candidates.sort(key=lambda x: x[2])  # Sort by eviction score

        # Progressive eviction - start with lowest scoring models
        for mid, entry, score, mem_gb in candidates:
            self.logger.info(
                f"🎯 Evicting {mid} (score: {score:.2f}, mem: {mem_gb:.2f}GB, priority: {entry.priority.name})"
            )
            with self._lock:
                removed = self._cache.pop(mid, None)
            if removed and removed.pipeline:
                self._cleanup_pipeline(removed.pipeline)
            hardware_manager.clear_memory()

            if hardware_manager.check_memory_available(required):
                self.logger.info(f"✅ Memory freed after evicting {mid}, proceeding")
                return True

        # If we still don't have enough memory, consider evicting protected models as last resort
        if protected and not hardware_manager.check_memory_available(required):
            self.logger.warning(
                "⚠️ Still insufficient memory, considering evicting protected models as last resort"
            )
            # Sort protected by score and evict the lowest scoring ones
            protected.sort(key=lambda x: x[2])  # Sort by eviction score

            for mid, mem_gb, score in protected[
                :2
            ]:  # Only evict up to 2 protected models
                self.logger.warning(
                    f"🚨 Last resort: evicting protected model {mid} (score: {score:.2f}, mem: {mem_gb:.2f}GB)"
                )
                with self._lock:
                    removed = self._cache.pop(mid, None)
                if removed and removed.pipeline:
                    self._cleanup_pipeline(removed.pipeline)
                hardware_manager.clear_memory()

                if hardware_manager.check_memory_available(required):
                    self.logger.info(
                        f"✅ Memory freed after protected eviction of {mid}"
                    )
                    return True

        final_available = hardware_manager.check_memory_available(required)
        if not final_available:
            self.logger.error(
                f"❌ Could not free sufficient memory for {required/1e9:.2f}GB model after all eviction attempts"
            )
        return final_available

    # ---- Background cleanup ----
    def _start_cleanup_thread(self) -> None:
        if self._cleanup_thread and self._cleanup_thread.is_alive():  # pragma: no cover
            return
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="LocalPipelineCacheCleanup"
        )
        self._cleanup_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the cleanup thread to stop and force-clean the cache.

        This is safe to call multiple times.
        """
        try:
            self._stop_event.set()
            if self._cleanup_thread and self._cleanup_thread.is_alive():
                self._cleanup_thread.join(timeout=timeout)
        except Exception:
            pass
        # Ensure any remaining pipelines are cleaned up
        try:
            self.force_cleanup()
        except Exception:
            pass

    def _cleanup_loop(self) -> None:  # pragma: no cover
        while not self._stop_event.is_set():
            try:
                # Wake periodically or when stopped

                # Sleep in small intervals so stop() is responsive
                for _ in range(6):
                    if self._stop_event.is_set():
                        break
                    time.sleep(10)
                if self._stop_event.is_set():
                    break
                self.clear_expired()

                # Periodic GPU thermal check
                try:
                    hardware_manager.check_gpu_thermals()
                except Exception:
                    pass
            except Exception:
                pass

    def _cleanup_pipeline(self, pipeline: BasePipeline | Embeddings) -> None:
        """Properly cleanup pipeline resources."""
        try:
            if hasattr(pipeline, "server_manager"):
                del pipeline.server_manager  # type: ignore[attr-defined]
            del pipeline

        except Exception as e:
            self.logger.warning(f"Error during pipeline cleanup: {e}")


# Module-global cache manager for consumers that want a shared cache instance.
# This allows external services to force cleanup on startup/shutdown.
try:
    local_pipeline_cache: LocalPipelineCacheManager = LocalPipelineCacheManager()
except Exception:
    # Fallback: if construction fails (e.g., during import in restricted env), create a minimal instance later
    local_pipeline_cache = None  # type: ignore
