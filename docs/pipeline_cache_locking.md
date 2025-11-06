# Pipeline Cache Locking

## Overview

This document describes the pipeline cache locking system that prevents eviction of pipelines during active inference operations. The system provides automatic protection against race conditions where pipelines could be evicted while generating responses.

## Core Architecture

### Pipeline Cache Entry Locking

Each pipeline cache entry (`_PipelineCacheEntry`) includes:

- `lock_count: int` - Number of active locks (supports nested/concurrent locking)
- `in_use: bool` - Computed property indicating if pipeline is locked

```python
# Lock/unlock operations support concurrent usage
entry.lock()    # Increment lock count
entry.unlock()  # Decrement lock count (with underflow protection)
```

### Automatic Eviction Protection

The cache manager automatically skips locked pipelines during eviction:

```python
def _ensure_memory(self, model_profile):
    # Eviction logic automatically respects locks
    candidates = [
        entry for entry in self._cache.values()
        if not entry.in_use  # Skip locked pipelines
    ]
```

### PipelineFactory Integration

The `PipelineFactory` provides automatic locking with a clean, safe API:

```python
# get_pipeline() automatically locks local pipelines
pipeline = factory.get_pipeline(profile)  # Auto-locked for local models

# Context manager provides automatic cleanup
with factory.pipeline(profile) as pipeline:
    # Pipeline is automatically locked (if local)
    response = await pipeline.generate(prompt)
    # Pipeline is automatically unlocked on exit
```

## Usage Patterns

### Basic Usage (Recommended)

The simplest and safest approach uses the context manager:

```python
from runner.pipeline_factory import PipelineFactory

# Context manager handles all locking automatically
with factory.pipeline(profile) as pipeline:
    # Safe to use pipeline - protected from eviction
    response = await pipeline.generate(prompt)
    # Automatic unlock when done
```

### Manual Pipeline Management

For advanced use cases requiring manual control:

```python
# Manual locking (local pipelines only)
pipeline = factory.get_pipeline(profile)  # Auto-locked if local
try:
    response = await pipeline.generate(prompt)
finally:
    factory.unlock_pipeline(profile)  # Manual cleanup required
```

### Remote vs Local Pipelines

The locking system intelligently handles different pipeline types:

```python
# Local pipelines: Automatic locking applied
local_profile = ModelProfile(model_id="llama-8b-local")
with factory.pipeline(local_profile) as pipeline:
    # Pipeline is locked and protected from eviction
    pass

# Remote pipelines: No locking needed (not cached locally)  
remote_profile = ModelProfile(model_id="gpt-4")
with factory.pipeline(remote_profile) as pipeline:
    # No locking overhead - remote pipelines aren't cached
    pass
```

## Implementation Details

### LocalPipelineCacheManager

Provides both manual and context manager APIs:

```python
# Manual operations
success = cache_manager.lock_pipeline(model_id)
cache_manager.unlock_pipeline(model_id)

# Context manager (legacy - prefer PipelineFactory)
with cache_manager.pipeline_in_use(model_id) as locked:
    if locked:
        # Pipeline successfully locked
        pass
```

### PipelineFactory Design

The factory coordinates between local cache locking and usage tracking:

```python
def get_pipeline(self, profile):
    """Get pipeline with automatic locking for local models."""
    pipeline = self.create_pipeline(profile)
    
    # Auto-lock local pipelines after creation
    if self.local_cache.is_local(profile.model_id):
        self.local_cache.lock_pipeline(profile.model_id)
        
    return pipeline
```

## Thread Safety

The locking system is thread-safe with proper coordination:

- Cache entry locks use atomic operations
- Factory tracks active uses with thread-safe counters
- Context managers ensure proper cleanup even on exceptions

## Performance Considerations

- **Negligible overhead**: Lock operations are simple counter increments
- **Local-only locking**: Remote pipelines skip locking entirely
- **Efficient eviction**: Locked pipeline detection is O(1) per cache entry

## Best Practices

1. **Use context managers**: Automatic cleanup prevents lock leaks
2. **Prefer factory methods**: Higher-level APIs handle complexities
3. **Trust automatic behavior**: The system intelligently handles local vs remote
4. **Monitor lock counts**: Use cache stats for debugging if needed

## Error Handling

The system includes comprehensive error handling:

- Underflow protection prevents negative lock counts
- Missing pipeline graceful handling  
- Exception safety in context managers
- Debug logging for troubleshooting

## Cache Statistics

Lock information is included in cache statistics:

```python
stats = cache_manager.stats()
# {
#     "total_entries": 3,
#     "locked_entries": 1,  # Number of currently locked pipelines
#     "lock_details": {
#         "model-1": {"lock_count": 2, "in_use": True}
#     }
# }
```
