# Simplify Pipeline Cache — Design Spec

## Problem

The pipeline loading, caching, and eviction system spans three files (~1300 lines) and is overly complex:

- **Complex eviction scoring** — priority bonuses, access frequency, memory efficiency, age penalties. Hard to reason about, calculations are off, evictions happen when they shouldn't.
- **Auto-persistence** — small models (<2GB) are auto-marked persistent, adding invisible behavior.
- **Dynamic timeouts** — multiplier chains based on size, priority, access count. Unpredictable.
- **Aggressive hardware management** — CUDA context destruction, defragmentation, process killing, thermal checks, power capping. Most of it unnecessary and error-prone.
- **Inflated memory estimation** — Resizer + OptimalParameters + memory breakdown formulas that don't match reality. The GGUF file size is already a known, accurate number.
- **Background cleanup thread** — adds complexity (stop events, join timeouts) for little benefit.

## Goal

Reduce the system to ~300 lines total that are simple, predictable, and correct:

1. Cache local models; return from cache when available.
2. Never evict a model that is actively in use.
3. Evict idle models only when VRAM is needed for an incoming model.
4. Proactively evict models that have been idle too long.
5. Fail the request (don't guess) when there's genuinely not enough VRAM.

## Configuration

Two environment variables, read via `server/config.py`:

| Variable | Default | Description |
|---|---|---|
| `PIPELINE_CACHE_TIMEOUT_MIN` | `30` | Idle models become *eviction candidates*. Only evicted when VRAM is needed for an incoming model. |
| `PIPELINE_EVICTION_TIMEOUT_MIN` | `60` | Hard expiry. Models idle longer than this are evicted proactively, regardless of VRAM pressure. |

Both are integers (minutes). Both checked on every `get()` call and in the background cleanup loop.

## Architecture

### PipelineCache (replaces `LocalPipelineCacheManager`)

A single class, ~150 lines. Holds a `Dict[str, _CacheEntry]` where `_CacheEntry` stores the pipeline, last-accessed timestamp, and use-count.

```python
class PipelineCache:
    def __init__(self):
        self._cache: Dict[str, _CacheEntry] = {}
        self._lock = threading.RLock()
        # Starts background thread for proactive eviction

    def get(self, model, create_fn) -> BasePipeline | Embeddings:
        """Return cached pipeline, or create and cache one."""
        # 1. Proactively evict any entries past EVICTION_TIMEOUT
        # 2. If in cache, return it (touch, increment use_count)
        # 3. Estimate size = model.details.size + 128MB overhead
        # 4. If available_vram < estimated_size:
        #    a. Evict idle candidates (idle > CACHE_TIMEOUT, oldest first, skip locked)
        #    b. If still not enough, raise InsufficientVRAMError
        # 5. Create, cache, return

    def unlock(self, model_id: str) -> bool:
        """Decrement use_count. Prevents eviction while > 0."""

    def clear(self, model_id: Optional[str] = None) -> None:
        """Remove specific entry, or all entries."""

    def stats(self) -> Dict:
        """Return count, entries (with size, in_use, last_accessed), and vram usage."""

    def stop(self) -> None:
        """Stop background thread, clear all entries. Called on shutdown."""
```

#### `_CacheEntry`

```python
@dataclass
class _CacheEntry:
    pipeline: BasePipeline | Embeddings
    last_accessed: float          # time.time()
    use_count: int                 # lock counter
    estimated_size_bytes: float    # model.details.size + overhead
```

### InsufficientVRAMError (new exception)

```python
class InsufficientVRAMError(Exception):
    def __init__(self, required_bytes: float, loaded_models: List[Dict]):
        # loaded_models = [{"name": "foo", "size_gb": 4.2, "in_use": True}, ...]
        self.required_bytes = required_bytes
        self.loaded_models = loaded_models
        msg = (f"Insufficient VRAM for model requiring {required_bytes/1e9:.1f}GB. "
               f"Loaded models: {', '.join(m['name'] for m in loaded_models)}. "
               f"Try again later.")
        super().__init__(msg)
```

The API layer catches this and returns a 503 with the loaded-models list in the response body.

### HardwareManager (simplified)

Strip `hardware_manager.py` to ~80 lines. Two methods:

```python
class HardwareManager:
    def available_vram_bytes(self) -> float:
        """Total free VRAM across all GPUs, in bytes."""

    def gpu_stats(self) -> Dict[str, Dict]:
        """Per-GPU stats: name, total_mb, used_mb, free_mb."""
```

Uses `nvsmi` (already a dependency) to query GPU memory. No CUDA context destruction, no defragmentation, no process killing, no thermal checks, no power capping.

Removes: `CUDAContextManager`, `GPUProcessManager`, `MemoryManager`, `MemoryConfig`, all thermal/power logic.

### PipelineFactory (simplified)

~100 lines. Responsibilities:

1. Route local vs remote providers (unchanged).
2. For local: delegate to `PipelineCache.get()`.
3. For remote: create transient pipeline (unchanged).
4. `create_pipeline()` — task-based routing to `_create_text_pipeline`, `_create_embedding_pipeline`, etc. (unchanged).

Removes: `_coord_lock`, `_coord_cond`, `_active_loads`, `_active_local_uses`, `pipeline()` context manager (zero external callers), `set_pipeline_persistent`, `force_evict_pipeline`, `get_cache_stats` (delegate to cache.stats).

### Memory Estimation

**Use `model.details.size` + 128MB overhead.**

`model.details.size` is the GGUF file size in bytes — the actual weight data loaded into VRAM. The 128MB overhead covers KV cache structures and runtime state. This is simpler and more accurate than the current Resizer-based formula.

No Resizer, no OptimalParameters, no memory breakdown.

### Background Thread

A single daemon thread that runs every 60 seconds:
1. Evict any entries past `PIPELINE_EVICTION_TIMEOUT_MIN` (hard expiry, skip locked).
2. That's it. No thermal checks, no GPU process info gathering.

### What Stays the Same

- Local vs remote provider routing (llama.cpp = cached, OpenAI/Anthropic = transient)
- `BasePipeline` interface and all pipeline implementations
- Server startup/shutdown lifecycle (`stop()` on shutdown)
- Lock/unlock pattern to prevent eviction during active generation
- Module-level singleton `pipeline_cache` for shared access

### Files Changed

| File | Action |
|---|---|
| `inference/runner/pipeline_cache.py` | Rewrite — `PipelineCache` + `_CacheEntry`, ~150 lines |
| `inference/runner/pipeline_factory.py` | Simplify — remove context manager, coordination locks, delegation methods, ~100 lines |
| `inference/runner/utils/hardware_manager.py` | Strip — `HardwareManager` with `available_vram_bytes()` + `gpu_stats()`, ~80 lines |
| `inference/server/config.py` | Add `PIPELINE_CACHE_TIMEOUT_MIN` and `PIPELINE_EVICTION_TIMEOUT_MIN` env vars |
| `inference/runner/exceptions.py` | New — `InsufficientVRAMError` |
| `inference/test/unit/test_pipeline_cache_locking.py` | Update tests for new API |

### Caller Impact

All callers use `pipeline_factory.get_pipeline()` or `pipeline_factory.get_embedding_pipeline()` — both still exist with the same signature. No caller changes needed except:

- `cleanup_for_user()` call in `server/routers/conversation.py` → replace with `pipeline_cache.clear()`
- `local_pipeline_cache.stop()` call in `server/app.py` → `pipeline_cache.stop()`
- `pipeline_factory.get_cache_stats()` → `pipeline_cache.stats()`
