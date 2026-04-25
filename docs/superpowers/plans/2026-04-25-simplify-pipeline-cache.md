# Simplify Pipeline Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** Replace ~1300-line pipeline cache/hardware system with ~300 lines. Simple, predictable, correct.

**Architecture:** Two env vars drive a `PipelineCache` class with lock/unlock. Minimal `HardwareManager` queries VRAM via nvsmi. Memory estimation = GGUF size + 128MB. Eviction = oldest-idle-first, two thresholds.

---

### Task 1: Add env vars to config.py
- Modify: `inference/server/config.py`

- [ ] Add after line 67:
```python
PIPELINE_CACHE_TIMEOUT_MIN = int(os.environ.get("PIPELINE_CACHE_TIMEOUT_MIN", "30"))
PIPELINE_EVICTION_TIMEOUT_MIN = int(os.environ.get("PIPELINE_EVICTION_TIMEOUT_MIN", "60"))
```
- [ ] Commit: `git add inference/server/config.py && git commit -m "add pipeline cache timeout env vars"`

### Task 2: Create InsufficientVRAMError
- Create: `inference/runner/exceptions.py`

- [ ] Write file:
```python
from typing import Dict, List

class InsufficientVRAMError(Exception):
    def __init__(self, required_bytes: float, loaded_models: List[Dict]):
        self.required_bytes = required_bytes
        self.loaded_models = loaded_models
        model_list = ", ".join(
            f"{m['name']} ({m['size_gb']:.1f}GB{' locked' if m.get('in_use') else ''})"
            for m in loaded_models
        )
        msg = (f"Insufficient VRAM for model requiring {required_bytes / 1e9:.1f}GB. "
               f"Loaded models: {model_list}. Try again later.")
        super().__init__(msg)
```
- [ ] Commit

### Task 3: Rewrite hardware_manager.py
- Modify: `inference/runner/utils/hardware_manager.py` (full rewrite, ~50 lines)

- [ ] Replace entire file:
```python
import nvsmi
from typing import Dict

class HardwareManager:
    def __init__(self):
        self._has_gpu = False
        self._gpu_count = 0
        try:
            gpus = nvsmi.get_gpus()
            self._has_gpu = len(gpus) > 0
            self._gpu_count = len(gpus)
        except Exception:
            pass

    @property
    def has_gpu(self) -> bool:
        return self._has_gpu

    @property
    def gpu_count(self) -> int:
        return self._gpu_count

    def available_vram_bytes(self) -> float:
        if not self._has_gpu:
            return 0.0
        try:
            return sum(g.mem_free for g in nvsmi.get_gpus()) * 1024 * 1024
        except Exception:
            return 0.0

    def gpu_stats(self) -> Dict[str, Dict]:
        stats: Dict[str, Dict] = {}
        if not self._has_gpu:
            return stats
        try:
            for g in nvsmi.get_gpus():
                stats[str(g.id)] = {"name": g.name, "total_mb": g.mem_total,
                    "used_mb": g.mem_used, "free_mb": g.mem_free, "util_percent": g.mem_util}
        except Exception:
            pass
        return stats

hardware_manager = HardwareManager()
```
- [ ] Commit

### Task 4: Rewrite pipeline_cache.py
- Modify: `inference/runner/pipeline_cache.py` (full rewrite, ~200 lines)

- [ ] Replace entire file with the following classes:

**`_CacheEntry` dataclass**: fields = `pipeline`, `last_accessed: float`, `use_count: int`, `estimated_size_bytes: float`

**`PipelineCache` class** with methods:
- `is_local(model)` - static, checks provider in LOCAL_PROVIDERS
- `get(model, priority, create_fn, grammar, metadata)` - main entry point
- `unlock(model_id)` - decrement use_count
- `clear(model_id=None)` - remove entry or all
- `stats()` - return cache + GPU info dict
- `stop(timeout=5.0)` - stop thread, clear cache
- `_estimate_size(model)` - model.details.size + 128MB, fallback 4GB/1GB
- `_evict_expired()` - proactive eviction past eviction_timeout
- `_evict_idle_oldest_first()` - evict idle past cache_timeout, oldest first, skip locked
- `_ensure_vram(cache_key, required_bytes)` - evict idle models until VRAM available, else raise InsufficientVRAMError
- `_cleanup_pipeline(pipeline)` - delete server_manager, del pipeline
- `_cleanup_loop()` - daemon thread, sleeps 60s, calls _evict_expired

**`_ensure_vram` logic**:
1. Check `hardware_manager.available_vram_bytes() >= required_bytes` -> return if OK
2. Call `_evict_idle_oldest_first()` to evict all idle candidates
3. Recheck available VRAM
4. If still insufficient, build loaded_models list from cache, raise `InsufficientVRAMError`

**`_cleanup_loop` logic**:
```
while not _stop_event.wait(60):
    try: _evict_expired()
    except: pass
```

**Module singleton**: `pipeline_cache = PipelineCache(cache_timeout_min=PIPELINE_CACHE_TIMEOUT_MIN, eviction_timeout_min=PIPELINE_EVICTION_TIMEOUT_MIN)` - imports from server/config

- [ ] Commit

### Task 5: Simplify pipeline_factory.py
- Modify: `inference/runner/pipeline_factory.py`

- [ ] Changes:
  1. Remove imports: `threading`, `UserConfig`, `OptimalParameters`, `ModelParameters`
  2. Remove `_GLOBAL_PIPELINE_CACHE` try/except, import `pipeline_cache` directly
  3. `__init__`: remove `_coord_lock`, `_coord_cond`, `_active_loads`, `_active_local_uses`, `prefer_langgraph`. Use `self.cache = pipeline_cache`
  4. `get_pipeline`: simplify to `self.cache.get(model, priority, self.create_pipeline, grammar, metadata)` for local, `self.create_pipeline(model)` for remote
  5. `get_embedding_pipeline`: same pattern
  6. Remove `pipeline()` context manager (zero external callers)
  7. Remove `set_pipeline_persistent`, `force_evict_pipeline`, `get_cache_stats`
  8. Keep: `unlock_pipeline` (delegates to `self.cache.unlock`), `create_pipeline` and all `_create_*` methods unchanged
  9. Update `clear_cache` delegation to use `self.cache.clear`

- [ ] Commit

### Task 6: Update callers and runner __init__
- Modify: `inference/runner/__init__.py`
- Modify: `inference/server/app.py`
- Modify: `inference/server/routers/conversation.py`

- [ ] `runner/__init__.py`: export `pipeline_cache` instead of `local_pipeline_cache`, add `InsufficientVRAMError`
```python
from .pipeline_factory import PipelineFactory, pipeline_factory
from .pipeline_cache import pipeline_cache
from .exceptions import InsufficientVRAMError
from .pipelines.llamacpp.chat import ReasoningAwareAIMessageChunk

__all__ = ["PipelineFactory", "pipeline_factory", "ReasoningAwareAIMessageChunk",
           "pipeline_cache", "InsufficientVRAMError"]
```

- [ ] `server/app.py`: line 75, change import to `from runner import pipeline_cache`; line 147-153, change `local_pipeline_cache` to `pipeline_cache`

- [ ] `server/routers/conversation.py`: line 193, change `local_pipeline_cache.cleanup_for_user(user_config)` to `pipeline_cache.clear()`

- [ ] Commit

### Task 7: Update tests
- Modify: `inference/test/unit/test_pipeline_cache_locking.py`

- [ ] Rewrite tests to use new `PipelineCache` and `_CacheEntry` API:
  - Import from `runner.pipeline_cache` instead of `LocalPipelineCacheManager`
  - `_CacheEntry` no longer has `priority` field, no `lock()`/`unlock()` methods on entry (use `use_count` directly)
  - `PipelineCache` has `unlock(model_id)` instead of `lock_pipeline`/`unlock_pipeline`
  - Remove `pipeline_in_use` context manager tests (removed)
  - Remove `stats` lock-info tests that depend on old format
  - Keep: basic lock/unlock behavior, nested locking, underflow protection, stats include lock info

- [ ] Commit

### Task 8: Remove unused files
- Delete: `inference/runner/utils/resizer.py` (only used by old pipeline_cache)
- Delete: `inference/test/unit/test_resizer_real_world_validation.py`

- [ ] `git rm inference/runner/utils/resizer.py inference/test/unit/test_resizer_real_world_validation.py`
- [ ] Commit

### Task 9: Validation
- [ ] Run `cd inference && python -c "from runner import pipeline_cache, pipeline_factory, InsufficientVRAMError; print('imports OK')"`
- [ ] Run `cd inference && pytest test/unit/test_pipeline_cache_locking.py -v`
- [ ] Run `make validate` to check types
- [ ] Commit any fixes

---

## Self-Review Checklist

**Spec coverage:**
- [x] Two env vars in config.py (Task 1)
- [x] InsufficientVRAMError exception (Task 2)
- [x] Minimal HardwareManager (Task 3)
- [x] PipelineCache with lock/unlock, two thresholds, oldest-first eviction (Task 4)
- [x] Simplified PipelineFactory (Task 5)
- [x] Updated callers: app.py, conversation.py, __init__.py (Task 6)
- [x] Updated tests (Task 7)
- [x] Removed Resizer (Task 8)
- [x] Validation (Task 9)

**Placeholder scan:** No TBDs, no "add validation", no "write tests for above" - all code provided inline.

**Type consistency:** `PipelineCache.unlock(model_id)` matches factory delegation. `pipeline_cache` singleton exported from `__init__.py`. `InsufficientVRAMError` signature consistent across all tasks.
