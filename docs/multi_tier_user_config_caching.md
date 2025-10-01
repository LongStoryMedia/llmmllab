# Multi-Tier User Configuration Caching System

## Overview

The LLM ML Lab platform implements a sophisticated **three-tier caching system** for user configuration data to optimize performance and reduce database load:

```text
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   In-Memory     │ -> │      Redis       │ -> │    Database      │
│   Cache         │    │      Cache       │    │   (PostgreSQL)   │
│  (5 min TTL)    │    │   (30 min TTL)   │    │  (Authoritative) │
└─────────────────┘    └──────────────────┘    └──────────────────┘
    Tier 1               Tier 2                  Tier 3
```

## Architecture Features

### Tier 1: In-Memory Cache

- **Storage**: Local Python dictionary with threading locks
- **TTL**: 5 minutes (configurable)
- **Capacity**: 500 entries (LRU eviction)
- **Performance**: Fastest retrieval (~0.001ms)
- **Scope**: Per-process instance

### Tier 2: Redis Cache

- **Storage**: Redis key-value store
- **TTL**: 30 minutes (configurable via Redis)
- **Capacity**: Limited by Redis memory configuration
- **Performance**: Fast retrieval (~1-5ms)
- **Scope**: Shared across all service instances

### Tier 3: Database (PostgreSQL)

- **Storage**: Persistent PostgreSQL database
- **TTL**: Permanent storage
- **Capacity**: No practical limit
- **Performance**: Slower retrieval (~10-50ms)
- **Scope**: Authoritative source of truth

## Implementation Details

### UserConfigStorage Class

The `UserConfigStorage` class implements the multi-tier caching system with the following key methods:

#### Retrieval Flow (`get_user_config`)

```python
async def get_user_config(self, user_id: str) -> UserConfig:
    # Tier 1: Check in-memory cache
    config = self._memory_cache_get(user_id)
    if config:
        return config  # Fast path
    
    # Tier 2: Check Redis cache
    config = cache_storage.get_user_config_from_cache(user_id)
    if config:
        self._memory_cache_set(user_id, config)  # Populate Tier 1
        return config
    
    # Tier 3: Load from database
    config = await self._get_user_config_from_database(user_id)
    if config:
        self._memory_cache_set(user_id, config)      # Populate Tier 1
        cache_storage.cache_user_config(user_id, config)  # Populate Tier 2
        return config
```

#### Update Flow (`update_user_config`)

```python
async def update_user_config(self, user_id: str, cfg: UserConfig) -> None:
    # Update authoritative database first
    await self._update_user_config_in_database(user_id, cfg)
    
    # Invalidate all cache tiers
    self.invalidate_user_config_cache(user_id)
    
    # Repopulate all cache tiers with updated config
    self._memory_cache_set(user_id, cfg)
    cache_storage.cache_user_config(user_id, cfg)
```

### In-Memory Cache Implementation

The in-memory cache uses a simple but effective design:

```python
cache_structure = {
    'cache': {},         # user_id -> (config, expires_at, access_count)
    'max_size': 500,     # LRU eviction threshold
    'default_ttl': 300,  # 5 minutes in seconds
    'hits': 0,          # Performance metrics
    'misses': 0,        # Performance metrics
    'lock': threading.RLock()  # Thread safety
}
```

#### Key Features

1. **TTL Expiration**: Entries automatically expire after the configured TTL
2. **LRU Eviction**: Oldest entries are removed when capacity is reached
3. **Thread Safety**: Uses `threading.RLock()` for concurrent access
4. **Statistics**: Tracks hits, misses, and hit rates for monitoring
5. **Lazy Initialization**: Cache is created only when first accessed

## Configuration Options

### TTL Settings

```python
# Default TTL values (configurable)
MEMORY_CACHE_TTL = 300     # 5 minutes
REDIS_CACHE_TTL = 1800     # 30 minutes
```

### Capacity Limits

```python
# Default capacity settings
MEMORY_CACHE_MAX_SIZE = 500   # Maximum entries in memory
# Redis capacity is limited by available memory
```

## Cache Invalidation Strategy

### Write-Through Invalidation

When user configuration is updated:

1. **Update Database**: Authoritative source is updated first
2. **Invalidate All Tiers**: Remove stale entries from all cache layers
3. **Repopulate Caches**: Store the updated configuration in all tiers

### Manual Invalidation

```python
# Invalidate specific user
storage.invalidate_user_config_cache(user_id)

# Clear all memory cache entries (for testing/debugging)
storage.clear_memory_cache()
```

## Performance Benefits

### Cache Hit Scenarios

| Scenario | Tier Hit | Response Time | Database Queries |
|----------|----------|---------------|------------------|
| Memory Hit | Tier 1 | ~0.001ms | 0 |
| Redis Hit | Tier 2 | ~1-5ms | 0 |
| Database Hit | Tier 3 | ~10-50ms | 1 |

### Expected Hit Rates

- **Memory Cache**: 70-90% (frequent users, recent activity)
- **Redis Cache**: 85-95% (distributed across service instances)
- **Database**: 100% (authoritative source)

## Monitoring and Statistics

### Cache Statistics API

```python
stats = storage.get_cache_stats()
```

Returns comprehensive metrics:

```python
{
    "memory_cache": {
        "enabled": true,
        "entries": 245,
        "max_size": 500,
        "hits": 1547,
        "misses": 203,
        "hit_rate": 0.884,
        "total_requests": 1750
    },
    "redis_cache": {
        "enabled": true,
        "connected": true
    },
    "database": {
        "enabled": true
    }
}
```

### Key Metrics to Monitor

1. **Hit Rate**: Should be >80% for memory cache, >90% for Redis
2. **Cache Size**: Monitor memory usage and eviction rates
3. **Miss Patterns**: High miss rates may indicate TTL tuning needed
4. **Response Times**: Track performance across all tiers

## Error Handling and Resilience

### Cache Failure Scenarios

1. **Memory Cache Unavailable**: Falls back to Redis → Database
2. **Redis Unavailable**: Falls back to Memory → Database
3. **Database Unavailable**: Returns cached data if available, else fails gracefully

### Thread Safety

All cache operations use proper locking to ensure thread safety:

- Memory cache uses `threading.RLock()`
- Redis operations are atomic
- Database connections use connection pooling

## Testing and Validation

### Unit Tests

The system includes comprehensive tests covering:

- Cache hit/miss scenarios
- TTL expiration behavior  
- LRU eviction logic
- Concurrent access patterns
- Invalidation strategies

### Performance Testing

Run the included test script to validate performance:

```bash
cd /path/to/inference
python test_cache_direct.py
```

## Configuration Management Integration

This multi-tier caching system integrates seamlessly with the composer configuration architecture:

1. **Composer Components** retrieve user configuration via `user_id`
2. **UserConfigStorage** handles all caching logic transparently
3. **Configuration Updates** automatically invalidate and refresh caches
4. **Performance** is optimized for frequent configuration access patterns

## Best Practices

### Development Guidelines

1. **Always Use `user_id`**: Never pass configuration objects between components
2. **Trust the Cache**: The system handles invalidation automatically  
3. **Monitor Performance**: Use cache statistics to optimize TTL settings
4. **Handle Failures Gracefully**: Implement fallbacks when caches are unavailable

### Production Considerations

1. **Memory Monitoring**: Watch memory usage of in-memory cache
2. **Redis Health**: Monitor Redis connectivity and performance
3. **Database Load**: Cache should significantly reduce database queries
4. **TTL Tuning**: Adjust TTL values based on usage patterns

## Future Enhancements

Potential improvements to consider:

1. **Distributed Memory Cache**: Share memory cache across service instances
2. **Cache Warming**: Preload frequently accessed configurations
3. **Smart Invalidation**: Selective invalidation based on configuration changes
4. **Compression**: Compress large configuration objects in Redis
5. **Metrics Dashboard**: Real-time cache performance visualization

This multi-tier caching system provides robust, performant user configuration management that scales with the platform's growth while maintaining data consistency and reliability.
