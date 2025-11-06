# Pipeline Cache Locking Usage Guide

The pipeline cache now includes locking functionality to prevent pipelines from being evicted while they're actively generating responses. This prevents race conditions that could interrupt ongoing inference operations.

## Pipeline Factory Integration (Recommended)

The easiest way to use safe pipeline locking is through the PipelineFactory, which automatically handles locking for local providers:

### 1. Context Manager (Preferred)

```python
from runner.pipeline_factory import pipeline_factory

profile = ModelProfile(
    user_id="user123",
    name="Chat Profile",
    model_name="qwen-model",
    parameters=ModelParameters(temperature=0.7),
    system_prompt="You are a helpful assistant",
    type=0
)

# Automatic locking for local providers, no locking for remote providers
with pipeline_factory.pipeline(profile) as pipeline:
    # Pipeline is automatically locked for local providers
    result = await pipeline.ainvoke(messages)
    # Automatic unlock when exiting context
```

### 2. Manual Safe Pipeline Usage

```python
# Get pipeline with automatic locking
pipeline = pipeline_factory.get_pipeline_safely(profile)
try:
    # Use the pipeline (protected from eviction if local)
    result = await pipeline.ainvoke(messages)
finally:
    # Unlock when done
    pipeline_factory.unlock_pipeline(profile)
```

## Direct Cache Manager Usage

For advanced use cases, you can work directly with the cache manager:

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

### 2. Context Manager for Cache

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

## Key Benefits

1. **Race Condition Prevention**: Pipelines cannot be evicted mid-generation
2. **Automatic Local/Remote Detection**: Factory automatically locks only local providers  
3. **Concurrent Usage Support**: Multiple requests can lock the same pipeline simultaneously  
4. **Automatic Cleanup**: Context managers ensure unlock even if exceptions occur
5. **Memory Pressure Awareness**: Eviction skips locked pipelines and logs warnings
6. **Observable State**: Cache statistics include lock counts for monitoring

## Architecture

- **Local Providers** (llama.cpp, stable_diffusion_cpp): Automatically locked to prevent eviction
- **Remote Providers** (OpenAI, etc.): No locking needed since they're not cached
- **Factory Integration**: Seamless locking without manual cache management
- **Memory Coordination**: Tracks active usage and coordinates eviction safely

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

1. **Use PipelineFactory context manager** - handles everything automatically
2. **Keep lock duration minimal** - only lock during actual inference
3. **Handle lock failures gracefully** - pipeline might not exist in cache
4. **Monitor lock counts** - high locked counts may indicate memory pressure
5. **Use appropriate priorities** - higher priority pipelines are less likely to be evicted

The integrated locking system ensures reliable inference operations even under memory pressure by protecting actively-used pipelines from eviction while maintaining optimal performance for remote providers.