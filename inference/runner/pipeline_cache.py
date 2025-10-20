"""Local pipeline cache & memory management for local model providers.

Extracted from pipeline_factory so only local (on-device) model providers
consume persistent cached resources. Remote/API providers bypass caching.
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
from typing import Any, Callable, Dict, List, Optional, Type, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

from models import Model, ModelProfile, ModelProvider, PipelinePriority
from .utils.hardware_manager import hardware_manager


class _PipelineCacheEntry:

    def __init__(
        self, pipeline: BaseChatModel | Embeddings, priority: PipelinePriority
    ):
        self._ref = weakref.ref(pipeline)
        self.priority = priority
        self.creation_time = time.time()
        self.last_accessed = self.creation_time
        self.access_count = 1

    @property
    def pipeline(self) -> Optional[BaseChatModel | Embeddings]:
        return self._ref()

    def is_alive(self) -> bool:
        return self._ref() is not None

    def touch(self) -> None:
        self.last_accessed = time.time()
        self.access_count += 1

    def eviction_score(self, now: float) -> float:
        age_penalty = (now - self.last_accessed) / 3600.0
        score = (
            float(self.priority.value)
            - age_penalty
            + min(self.access_count / 10.0, 2.0)
        )
        return score


class LocalPipelineCacheManager:
    """Caches pipelines only for local providers (llama.cpp, stable diffusion cpp)."""

    LOCAL_PROVIDERS = {ModelProvider.LLAMA_CPP, ModelProvider.STABLE_DIFFUSION_CPP}

    def __init__(self, cache_timeout: int = 300):
        self._cache: Dict[str, _PipelineCacheEntry] = {}
        self._lock = threading.RLock()
        self._cache_timeout = cache_timeout
        self.logger = logging.getLogger(__name__ + ".LocalCache")
        self._cleanup_thread: Optional[threading.Thread] = None
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
            [Model, ModelProfile, Optional[Type[BaseModel]]],
            Optional[BaseChatModel | Embeddings],
        ],
        grammar: Optional[Type[BaseModel]] = None,
    ) -> BaseChatModel | Embeddings:
        model_id = model.id or model.model
        with self._lock:
            entry = self._cache.get(model_id)
            if entry and entry.is_alive():
                entry.touch()
                pipe = entry.pipeline
                if pipe:
                    return pipe
            elif entry:
                self._cache.pop(model_id, None)

        required = self.estimate_memory(model, profile)
        if not self._ensure_memory(required, exclude=model_id):
            raise RuntimeError(
                f"Insufficient memory for local model {model.name}: need {required/1e9:.2f}GB"
            )

        pipeline = create_fn(model, profile, grammar)
        if not pipeline:
            raise RuntimeError(f"Failed to create pipeline for {model.name}")

        with self._lock:
            self._cache[model_id] = _PipelineCacheEntry(pipeline, priority)

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
        now = time.time()
        expired: List[str] = []
        with self._lock:
            for mid, entry in self._cache.items():
                if (
                    now - entry.last_accessed
                ) > self._cache_timeout or not entry.is_alive():
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
            mem = hardware_manager.update_all_memory_stats()
            return {
                "count": len(self._cache),
                "alive": len(alive),
                "dead": len(self._cache) - len(alive),
                "entries": {
                    mid: {
                        "priority": e.priority.name,
                        "access_count": e.access_count,
                        "last_accessed": e.last_accessed,
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

    def force_cleanup(self) -> int:
        with self._lock:
            count = len(self._cache)
            for mid, entry in list(self._cache.items()):
                self._cache.pop(mid, None)
                if entry and entry.pipeline:
                    self._cleanup_pipeline(entry.pipeline)
        hardware_manager.clear_memory(aggressive=True)
        return count

    # ---- Internals ----
    def estimate_memory(
        self, model: Model, profile: Optional["ModelProfile"] = None
    ) -> float:
        """Estimate memory usage based on model size and actual context parameters."""
        base = 512 * 1024 * 1024
        model_size = 0
        details = getattr(model, "details", None)

        if details and getattr(details, "parameter_size", None):
            try:
                raw = details.parameter_size.upper().strip()
                if raw.endswith("B"):
                    params = float(raw[:-1]) * 1_000_000_000
                elif raw.endswith("M"):
                    params = float(raw[:-1]) * 1_000_000
                elif raw.endswith("K"):
                    params = float(raw[:-1]) * 1_000
                else:
                    n = float(raw)
                    params = n * 1_000_000_000 if n > 1 else n * 1_000_000
                q = (details.quantization_level or "q4").lower()
                if "q4" in q or "iq4" in q:
                    bpp = 0.5
                elif "q5" in q:
                    bpp = 0.625
                elif "q6" in q:
                    bpp = 0.75
                elif "q8" in q:
                    bpp = 1.0
                elif any(x in q for x in ["fp16", "bf16", "f16"]):
                    bpp = 2.0
                else:
                    bpp = 4.0
                model_size = int(params * bpp)
            except Exception:  # noqa: BLE001
                pass
        if model_size == 0 and getattr(model, "size", 0):
            if model.size < 100 * 1024 * 1024 * 1024:
                model_size = model.size
        if model_size == 0:
            task = str(getattr(model, "task", "TextToText"))
            if task.endswith("TextToEmbeddings"):
                model_size = 1 * 1024 * 1024 * 1024
            elif task.endswith("TextToText"):
                model_size = 4 * 1024 * 1024 * 1024
            elif "Image" in task:
                model_size = 8 * 1024 * 1024 * 1024
            else:
                model_size = 2 * 1024 * 1024 * 1024

        # Calculate context memory based on actual context size from profile
        context_mem = model_size * 0.2  # Default 20% fallback
        if profile and profile.parameters and profile.parameters.num_ctx:
            # More accurate context memory calculation
            # Context memory scales with context size and model dimensions
            ctx_size = profile.parameters.num_ctx
            if ctx_size > 32768:  # Large context needs more memory
                context_multiplier = min(
                    ctx_size / 32768 * 0.3, 1.0
                )  # Cap at 100% of model size
                context_mem = model_size * context_multiplier
                self.logger.debug(
                    f"Large context detected ({ctx_size}), using {context_multiplier:.2f}x model size for context memory"
                )

        total = base + model_size + context_mem
        self.logger.debug(
            f"Memory estimate for {model.name}: {total/1e9:.2f}GB "
            f"(model: {model_size/1e9:.2f}GB, context: {context_mem/1e9:.2f}GB)"
        )
        return total

    def _ensure_memory(self, required: float, exclude: Optional[str]) -> bool:
        """Ensure sufficient memory is available, with aggressive upfront eviction for large models."""
        # For large models (>10GB), be proactive and clear cache immediately
        large_model = required > 10 * 1024 * 1024 * 1024  # 10GB threshold
        if large_model:
            self.logger.info(
                f"🚀 Large model detected ({required/1e9:.2f}GB), proactively clearing cache"
            )
            # Clear ALL other models immediately for large models
            with self._lock:
                evict_targets = [mid for mid in self._cache.keys() if mid != exclude]

            if evict_targets:
                self.logger.info(
                    f"🧹 Proactively evicting {len(evict_targets)} models: {evict_targets}"
                )
                for mid in evict_targets:
                    with self._lock:
                        removed = self._cache.pop(mid, None)
                    if removed and removed.pipeline:
                        self._cleanup_pipeline(removed.pipeline)

                # Nuclear memory clear after eviction
                hardware_manager.clear_memory(aggressive=True, nuclear=True)
                self.logger.info(
                    "🧹 Completed proactive cache clearing for large model"
                )

        # Check if we now have enough memory
        if hardware_manager.check_memory_available(required):
            return True

        # If still not enough, continue with standard eviction
        self.logger.info(
            f"Memory still low after proactive clearing; attempting additional eviction (need {required/1e9:.2f}GB)"
        )

        # Step 1: Clear dead entries
        with self._lock:
            dead = [mid for mid, e in self._cache.items() if not e.is_alive()]
            for mid in dead:
                self._cache.pop(mid, None)
        hardware_manager.clear_memory(aggressive=False)
        if hardware_manager.check_memory_available(required):
            return True

        # Step 2: Progressive eviction by priority
        now = time.time()
        with self._lock:
            candidates = [
                (mid, e, e.eviction_score(now))
                for mid, e in self._cache.items()
                if e.is_alive() and mid != exclude
            ]
        candidates.sort(key=lambda x: (x[2], x[1].priority, x[1].last_accessed))

        # Standard eviction - one at a time
        for mid, _, score in candidates:
            with self._lock:
                removed = self._cache.pop(mid, None)
            if removed and removed.pipeline:
                self._cleanup_pipeline(removed.pipeline)
            hardware_manager.clear_memory(aggressive=True)
            if hardware_manager.check_memory_available(required):
                self.logger.info(
                    f"Freed memory after evicting {mid} (score {score:.2f}); proceeding"
                )
                return True

        return hardware_manager.check_memory_available(required)

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
        try:
            cleanup_fn = getattr(pipeline, "cleanup", None)
            if callable(cleanup_fn):
                cleanup_fn()
            llm = getattr(pipeline, "llm", None)
            if llm is not None:
                llm_cleanup = getattr(llm, "cleanup", None)
                if callable(llm_cleanup):
                    llm_cleanup()
        except Exception:  # pragma: no cover
            pass
