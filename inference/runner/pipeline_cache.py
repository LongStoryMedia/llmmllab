"""Local pipeline cache & memory management for local model providers.

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

from models import (
    Model,
    ModelProfile,
    ModelProvider,
    PipelinePriority,
    UserConfig,
    OptimalParameters,
    ModelParameters,
)
from utils.logging import llmmllogger
from .utils.hardware_manager import hardware_manager
from .utils.resizer import Resizer
from .utils.intelligent_oom_recovery import IntelligentOOMRecovery


class _PipelineCacheEntry:

    def __init__(
        self,
        pipeline: BaseChatModel | Embeddings,
        priority: PipelinePriority,
        estimated_memory: float = 0,
    ):
        self._ref = weakref.ref(pipeline)
        self.priority = priority
        self.estimated_memory = (
            estimated_memory  # Store memory estimate for eviction decisions
        )
        self.creation_time = time.time()
        self.last_accessed = self.creation_time
        self.access_count = 1
        self.in_use = False  # Prevent eviction while pipeline is actively generating
        self._use_count = 0  # Track concurrent usage

    @property
    def pipeline(self) -> Optional[BaseChatModel | Embeddings]:
        return self._ref()

    @property
    def use_count(self) -> int:
        """Get the current use count for this pipeline."""
        return self._use_count

    def is_alive(self) -> bool:
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
            Optional[BaseChatModel | Embeddings],
        ],
        grammar: Optional[Type[BaseModel]] = None,
        user_config: Optional[UserConfig] = None,
        metadata: Optional[dict] = {},
    ) -> BaseChatModel | Embeddings:
        model_id = model.id or model.model
        with self._lock:
            entry = self._cache.get(model_id)
            if entry and entry.is_alive():
                entry.touch()
                pipe = entry.pipeline
                if pipe:
                    return pipe
            self._cache.pop(model_id, None)

        required = self.estimate_memory(model, profile)

        # Check if graceful degradation is enabled and try OOM recovery if needed
        if not self._ensure_memory(required, exclude=model_id):
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
                    if self._ensure_memory(optimized_required, exclude=model_id):
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
            self._cache[model_id] = _PipelineCacheEntry(pipeline, priority, required)
            self.logger.debug(f"💾 Cached NEW pipeline for {model_id}")

        # Auto-mark small models (likely embeddings) as persistent
        if required < 2 * 1024 * 1024 * 1024:  # < 2GB
            self.set_persistent(model_id, True)
            self.logger.info(f"🔒 Auto-marked small model {model_id} as persistent")

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

                # Calculate dynamic timeout based on pipeline characteristics
                base_timeout = self._cache_timeout

                # Small models get much longer timeout (they're cheap to keep)
                if entry.estimated_memory < 2 * 1024 * 1024 * 1024:  # < 2GB
                    timeout_multiplier = 10.0  # 10x longer timeout for small models
                elif entry.estimated_memory < 5 * 1024 * 1024 * 1024:  # < 5GB
                    timeout_multiplier = 3.0  # 3x longer for medium models
                else:
                    timeout_multiplier = 1.0  # Standard timeout for large models

                # High priority models get longer timeout
                if entry.priority.value >= 4:  # HIGH or URGENT priority
                    timeout_multiplier *= 2.0

                # Frequently accessed models get longer timeout
                if entry.access_count > 5:
                    timeout_multiplier *= 1.5

                dynamic_timeout = base_timeout * timeout_multiplier

                if (now - entry.last_accessed) > dynamic_timeout:
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
                # For persistent pipelines, significantly boost their eviction score
                if persistent:
                    # Set very high access count to make it less likely to be evicted
                    entry.access_count = max(entry.access_count, 1000)
                    # Update last accessed to prevent timeout
                    entry.touch()
                    self.logger.info(f"🔒 Marked pipeline {model_id} as persistent")
                else:
                    # Reset to normal access pattern
                    entry.access_count = min(entry.access_count, 10)
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
        hardware_manager.clear_memory(aggressive=True, nuclear=True)
        return count

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
                hardware_manager.clear_memory(aggressive=True, nuclear=True)
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
            hardware_manager.clear_memory(aggressive=False)
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
            hardware_manager.clear_memory(aggressive=True, nuclear=True)

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
                hardware_manager.clear_memory(aggressive=True, nuclear=True)

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

    def _cleanup_loop(self) -> None:  # pragma: no cover
        while True:
            try:
                time.sleep(60)
                self.clear_expired()
            except Exception:
                pass

    def _cleanup_pipeline(self, pipeline: BaseChatModel | Embeddings) -> None:
        """Properly cleanup pipeline resources by calling both close() and cleanup() methods."""
        try:
            # Try both close() and cleanup() methods as different pipelines use different names
            close_fn = getattr(pipeline, "close", None)
            if callable(close_fn):
                self.logger.debug(
                    f"Calling close() on pipeline {type(pipeline).__name__}"
                )
                close_fn()

            cleanup_fn = getattr(pipeline, "cleanup", None)
            if callable(cleanup_fn):
                self.logger.debug(
                    f"Calling cleanup() on pipeline {type(pipeline).__name__}"
                )
                cleanup_fn()

            # Also check nested llm attribute
            llm = getattr(pipeline, "llm", None)
            if llm is not None:
                llm_close = getattr(llm, "close", None)
                if callable(llm_close):
                    self.logger.debug(
                        f"Calling close() on nested llm {type(llm).__name__}"
                    )
                    llm_close()

                llm_cleanup = getattr(llm, "cleanup", None)
                if callable(llm_cleanup):
                    self.logger.debug(
                        f"Calling cleanup() on nested llm {type(llm).__name__}"
                    )
                    llm_cleanup()

            # Also check for llama_instance directly (BaseLlamaCppPipeline specific)
            llama_instance = getattr(pipeline, "llama_instance", None)
            if llama_instance is not None:
                llama_close = getattr(llama_instance, "close", None)
                if callable(llama_close):
                    self.logger.debug(
                        f"Calling close() on llama_instance {type(llama_instance).__name__}"
                    )
                    llama_close()

            self.logger.info(
                f"🗑️ Successfully cleaned up pipeline {type(pipeline).__name__}"
            )

        except Exception as e:
            self.logger.warning(f"Error during pipeline cleanup: {e}")
        finally:
            # Force deletion regardless of cleanup success
            try:
                if pipeline is not None:
                    del pipeline
            except Exception:
                pass
