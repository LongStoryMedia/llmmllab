# Pipeline Cache Locking Usage Guide

The pipeline cache now includes locking functionality to prevent pipelines from being evicted while they're actively generating responses. This prevents race conditions that could interrupt ongoing inference operations.

## Usage Patterns

### 1. Manual Locking/Unlocking

```python
from runner.pipeline_cache import LocalPipelineCacheManager

cache_manager = LocalPipelineCacheManager()

# Get your pipeline through normal means
pipeline = cache_manager.get_or_create(model, profile, priority, create_fn)
model_id = model.id or model.model

# Lock before starting inference
if cache_manager.lock_pipeline(model_id):
    try:
        # Safe to use pipeline - it won't be evicted
        result = await pipeline.generate(messages)
    finally:
        # Always unlock when done
        cache_manager.unlock_pipeline(model_id)
```

### 2. Context Manager (Recommended)

```python
# Automatic locking/unlocking with context manager
with cache_manager.pipeline_in_use(model_id) as locked:
    if locked:
        # Pipeline is locked and safe to use
        result = await pipeline.generate(messages)
        # Automatic unlock when exiting context
    else:
        # Pipeline not found or couldn't be locked
        handle_error()
```

### 3. Integration Example

```python
async def safe_chat_completion(model, messages):
    """Example of safe chat completion with pipeline locking."""
    
    # Get pipeline from cache
    pipeline = cache_manager.get_or_create(model, profile, priority, create_fn)
    model_id = model.id or model.model
    
    # Use context manager for automatic lock management
    with cache_manager.pipeline_in_use(model_id) as locked:
        if not locked:
            raise RuntimeError(f"Could not lock pipeline for {model_id}")
            
        # Pipeline is now protected from eviction
        try:
            return await pipeline.ainvoke(messages)
        except Exception as e:
            # Handle generation errors
            raise RuntimeError(f"Generation failed: {e}") from e
        # Pipeline automatically unlocked here
```

## Key Benefits

1. **Race Condition Prevention**: Pipelines cannot be evicted mid-generation
2. **Concurrent Usage Support**: Multiple requests can lock the same pipeline simultaneously  
3. **Automatic Cleanup**: Context manager ensures unlock even if exceptions occur
4. **Memory Pressure Awareness**: Eviction skips locked pipelines and logs warnings
5. **Observable State**: Cache statistics include lock counts for monitoring

## Monitoring

Check cache statistics to monitor locking behavior:

```python
stats = cache_manager.stats()
print(f"Total pipelines: {stats['count']}")
print(f"Currently locked: {stats['locked']}")

for model_id, entry_stats in stats['entries'].items():
    if entry_stats['in_use']:
        print(f"  {model_id}: locked (use_count={entry_stats['use_count']})")
```

## Best Practices

1. **Always use context manager** when possible for automatic cleanup
2. **Keep lock duration minimal** - only lock during actual inference
3. **Handle lock failures gracefully** - pipeline might not exist in cache
4. **Monitor lock counts** - high locked counts may indicate memory pressure
5. **Use appropriate priorities** - higher priority pipelines are less likely to be evicted

The locking system ensures reliable inference operations even under memory pressure by protecting actively-used pipelines from eviction.