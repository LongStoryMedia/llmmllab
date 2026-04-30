"""
Simple pipeline cache with two-threshold eviction for local model providers.

Only local (on-device) model providers consume cached resources.
Remote/API providers bypass caching entirely.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

from models import Model, ModelProvider, PipelinePriority
from runner.pipelines.base import BasePipeline
from utils.logging import llmmllogger
from .utils.hardware_manager import hardware_manager
from .exceptions import InsufficientVRAMError
from config import PIPELINE_CACHE_TIMEOUT_MIN, PIPELINE_EVICTION_TIMEOUT_MIN

_MODEL_OVERHEAD_BYTES = 128 * 1024 * 1024  # 128 MB


@dataclass
class _CacheEntry:
    pipeline: Union[BasePipeline, Embeddings]
    last_accessed: float = field(default_factory=time.time)
    use_count: int = 0
    estimated_size_bytes: float = 0


class PipelineCache:
    """Caches pipelines only for local providers (llama.cpp, stable diffusion cpp)."""

    LOCAL_PROVIDERS = {ModelProvider.LLAMA_CPP, ModelProvider.STABLE_DIFFUSION_CPP}

    def __init__(
        self,
        cache_timeout_min: int = 30,
        eviction_timeout_min: int = 60,
    ) -> None:
        self._cache: Dict[str, _CacheEntry] = {}
        self._lock = threading.RLock()
        self._cache_timeout_s = cache_timeout_min * 60
        self._eviction_timeout_s = eviction_timeout_min * 60
        self.logger = llmmllogger.logger.bind(component="PipelineCache")
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="PipelineCacheCleanup"
        )
        self._cleanup_thread.start()

    # ---- Public API ----

    @staticmethod
    def is_local(model: Model) -> bool:
        try:
            return model.provider in PipelineCache.LOCAL_PROVIDERS
        except Exception:
            return False

    def get(
        self,
        model: Model,
        priority: PipelinePriority,
        create_fn: Callable[
            [Model, Optional[Type[BaseModel]], Optional[dict]],
            Optional[Union[BasePipeline, Embeddings]],
        ],
        grammar: Optional[Type[BaseModel]] = None,
        metadata: Optional[dict] = None,
    ) -> Union[BasePipeline, Embeddings]:
        """Get a cached pipeline or create a new one.

        Proactively evicts expired entries, returns cached if exists
        (touching and incrementing use_count), otherwise creates new.
        Raises InsufficientVRAMError if not enough VRAM.
        """
        cache_key = model.name

        # Proactively evict expired entries
        self._evict_expired()

        with self._lock:
            entry = self._cache.get(cache_key)
            if entry is not None:
                pipe = entry.pipeline
                # Check server health
                sm = getattr(pipe, "server_manager", None)
                if sm is not None and hasattr(sm, "is_running") and not sm.is_running():
                        self.logger.warning(
                            f"Found cached pipeline for {cache_key} with dead server - evicting"
                        )
                        del self._cache[cache_key]
                        self._cleanup_pipeline(pipe)
                        entry = None  # continue to creation

                if entry is not None:
                    entry.last_accessed = time.time()
                    entry.use_count += 1
                    if isinstance(pipe, BasePipeline) and metadata:
                        pipe.bind_metadata(metadata)
                    self.logger.debug(f"Retrieved cached pipeline for {cache_key}")
                    return pipe

        # Create new pipeline
        self.logger.info(f"Creating new pipeline for {cache_key}")
        required_bytes = self._estimate_size(model)

        self._ensure_vram(cache_key, required_bytes)

        pipeline = create_fn(model, grammar, metadata)
        if pipeline is None:
            raise RuntimeError(f"Failed to create pipeline for {model.name}")

        with self._lock:
            entry = _CacheEntry(
                pipeline=pipeline,
                estimated_size_bytes=required_bytes,
                use_count=1,  # Creator holds the lock
            )
            self._cache[cache_key] = entry
            self.logger.debug(f"Cached new pipeline for {cache_key}")

        return pipeline

    def unlock(self, model_id: str) -> bool:
        """Decrement use_count for a model."""
        with self._lock:
            entry = self._cache.get(model_id)
            if entry is not None:
                entry.use_count = max(0, entry.use_count - 1)
                return True
        return False

    def clear(self, model_id: Optional[str] = None) -> None:
        """Remove a specific entry or all entries."""
        with self._lock:
            if model_id is not None:
                entry = self._cache.pop(model_id, None)
                if entry is not None:
                    self._cleanup_pipeline(entry.pipeline)
            else:
                entries = list(self._cache.values())
                self._cache.clear()
                for entry in entries:
                    self._cleanup_pipeline(entry.pipeline)

        self.logger.info(
            "Cleared %s pipeline cache entries",
            "all" if model_id is None else model_id,
        )

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        gpu = hardware_manager.gpu_stats()
        with self._lock:
            locked = sum(1 for e in self._cache.values() if e.use_count > 0)
            total_bytes = sum(e.estimated_size_bytes for e in self._cache.values())
            entries = {
                mid: {
                    "use_count": e.use_count,
                    "last_accessed": e.last_accessed,
                    "estimated_size_gb": e.estimated_size_bytes / 1e9,
                }
                for mid, e in self._cache.items()
            }

        return {
            "count": len(self._cache),
            "locked": locked,
            "total_cached_gb": total_bytes / 1e9,
            "entries": entries,
            "gpu": gpu,
        }

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background cleanup thread and clear all cached pipelines."""
        self._stop_event.set()
        if self._cleanup_thread is not None and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=timeout)
        try:
            self.clear()
        except Exception:
            pass

    # ---- Internals ----

    def _estimate_size(self, model: Model) -> float:
        """Estimate memory usage: GGUF file size + 128 MB overhead.

        Fallback: 4 GB for text models, 1 GB for embeddings.
        """
        try:
            size = model.details.size
            if size > 0:
                return size + _MODEL_OVERHEAD_BYTES
        except Exception:
            pass

        task = str(getattr(model, "task", ""))
        if "Embeddings" in task:
            return 1 * 1024 * 1024 * 1024  # 1 GB
        return 4 * 1024 * 1024 * 1024  # 4 GB

    def _evict_expired(self) -> None:
        """Remove entries past the eviction timeout. Skip locked (use_count > 0) entries."""
        now = time.time()
        to_evict: List[tuple] = []

        with self._lock:
            for mid, entry in self._cache.items():
                if entry.use_count > 0:
                    continue
                if (now - entry.last_accessed) > self._eviction_timeout_s:
                    to_evict.append((mid, entry.pipeline))

        for mid, pipe in to_evict:
            self.logger.debug(f"Evicting expired pipeline: {mid}")
            with self._lock:
                self._cache.pop(mid, None)
            self._cleanup_pipeline(pipe)

    def _evict_idle_oldest_first(self) -> List[str]:
        """Evict idle entries past the cache timeout, oldest first. Skip locked entries.

        Returns the list of evicted model IDs.
        """
        now = time.time()
        to_evict: List[tuple] = []

        with self._lock:
            idle = [
                (mid, entry)
                for mid, entry in self._cache.items()
                if entry.use_count == 0 and (now - entry.last_accessed) > self._cache_timeout_s
            ]
            # Sort by last_accessed ascending (oldest first)
            idle.sort(key=lambda x: x[1].last_accessed)
            to_evict = [(mid, entry.pipeline) for mid, entry in idle]
            for mid, _ in to_evict:
                self._cache.pop(mid, None)

        evicted_ids: List[str] = []
        for mid, pipe in to_evict:
            self.logger.debug(f"Evicting idle pipeline: {mid}")
            self._cleanup_pipeline(pipe)
            evicted_ids.append(mid)

        return evicted_ids

    def _ensure_vram(self, _cache_key: str, required_bytes: float) -> None:
        """Evict idle models until VRAM is available, else raise InsufficientVRAMError."""
        available = hardware_manager.available_vram_bytes()
        if available >= required_bytes:
            return

        self._evict_idle_oldest_first()

        if hardware_manager.available_vram_bytes() >= required_bytes:
            return

        # Build loaded_models list from current cache
        loaded: List[Dict[str, Any]] = []
        with self._lock:
            for mid, entry in self._cache.items():
                loaded.append(
                    {
                        "name": mid,
                        "size_gb": entry.estimated_size_bytes / 1e9,
                        "in_use": entry.use_count > 0,
                    }
                )

        raise InsufficientVRAMError(required_bytes, loaded)

    def _cleanup_pipeline(self, pipeline: Union[BasePipeline, Embeddings]) -> None:
        """Release pipeline resources."""
        try:
            if hasattr(pipeline, "server_manager"):
                del pipeline.server_manager  # type: ignore[attr-defined,reportAttributeAccessIssue]
            del pipeline
        except Exception as e:
            self.logger.warning(f"Error during pipeline cleanup: {e}")

    def _cleanup_loop(self) -> None:
        """Daemon thread: periodically evict expired entries."""
        while not self._stop_event.wait(60):
            try:
                self._evict_expired()
            except Exception:
                pass


# Module-global singleton
pipeline_cache = PipelineCache(
    cache_timeout_min=PIPELINE_CACHE_TIMEOUT_MIN,
    eviction_timeout_min=PIPELINE_EVICTION_TIMEOUT_MIN,
)

# Backward-compatible alias for existing import paths
local_pipeline_cache = pipeline_cache

# Deprecated class alias — remove once all callers migrate to PipelineCache
LocalPipelineCacheManager = PipelineCache
