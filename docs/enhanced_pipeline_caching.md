# Enhanced Pipeline Caching System

This document describes the intelligent pipeline caching system that keeps models in memory for as long as possible based on size, priority, and usage patterns.

## Overview

The enhanced pipeline caching system addresses the previous issue where pipelines were aggressively removed from memory after every completion. Now, pipelines remain cached and available for reuse based on intelligent eviction criteria that consider:

1. **Model size** (smaller models heavily favored)
2. **Priority level** (HIGH priority models protected)  
3. **Access frequency** (frequently used models stay longer)
4. **Memory pressure** (only evict when actually needed)

## Key Improvements

### 🎯 Size-Aware Eviction

Models are categorized by memory usage with different protection levels:

- **< 2GB (Small)**: Strong protection (5.0 bonus) + auto-persistence
- **2-5GB (Medium)**: Moderate protection (2.0 bonus)
- **5-10GB (Large)**: No bonus/penalty (0.0)
- **> 10GB (Very Large)**: Slight penalty (-2.0)

### 🛡️ Automatic Protection for Embeddings

Embedding models and other small models (< 2GB) are automatically marked as persistent and get:

- 10x longer timeout periods (50+ minutes vs 5 minutes)
- Highest eviction scores (protected unless absolutely necessary)
- Special tiny model bonus (< 1GB gets extra 0.1 score points)

### 📊 Enhanced Eviction Scoring

The eviction score determines which models to keep vs evict:

```python
score = priority_bonus + access_bonus + memory_bonus - age_penalty
```

- **Priority Bonus**: 2x priority value (HIGH=8, NORMAL=6, LOW=2)
- **Access Bonus**: Up to 3.0 based on usage frequency
- **Memory Bonus**: +5.0 for small, +2.0 for medium, -2.0 for very large
- **Age Penalty**: Increases over time (hours since last access)

### 🧠 Intelligent Memory Management

Memory eviction now follows progressive logic:

1. **Check available memory** - avoid unnecessary eviction
2. **For very large models (>15GB)** - aggressively clear space but protect small models
3. **Progressive eviction** - evict lowest scoring models first
4. **Protected model handling** - small valuable models avoided until last resort

### ⏱️ Dynamic Timeout System

Timeout periods are dynamically calculated based on model characteristics:

```python
timeout_multiplier = base_multiplier
if memory < 2GB: timeout_multiplier *= 10.0    # Small models: 10x longer
elif memory < 5GB: timeout_multiplier *= 3.0   # Medium models: 3x longer  
if priority >= HIGH: timeout_multiplier *= 2.0 # High priority: 2x longer
if access_count > 5: timeout_multiplier *= 1.5 # Frequent use: 1.5x longer
```

### 🔄 Agent Cleanup Improvements

Agents now use less aggressive cleanup:

- **Old behavior**: Unlock pipeline + clear reference → immediate availability for eviction
- **New behavior**: Unlock pipeline but keep cached → available for reuse by other agents

## Usage Examples

### Monitoring Cache Status

```python
from runner import pipeline_factory

# Get detailed cache statistics
stats = pipeline_factory.get_cache_stats()
print(f"Total models: {stats['total_models']}")
print(f"Small models: {stats['small_models']['count']} ({stats['small_models']['memory_gb']:.2f}GB)")
print(f"Large models: {stats['large_models']['count']} ({stats['large_models']['memory_gb']:.2f}GB)")
```

### Manual Persistence Control

```python
# Mark a specific pipeline as persistent (avoid eviction)
pipeline_factory.set_pipeline_persistent(profile, persistent=True)

# Remove persistence marking
pipeline_factory.set_pipeline_persistent(profile, persistent=False)

# Force eviction of specific pipeline
pipeline_factory.force_evict_pipeline(profile)
```

### Cache Manager Direct Access

```python
from runner.pipeline_cache import LocalPipelineCacheManager

cache = LocalPipelineCacheManager()

# Get detailed cache information
info = cache.get_cache_info()

# Manual persistence control
cache.set_persistent("model-id", True)

# View current cache stats
stats = cache.stats()
```

## Expected Behavior

### Embedding Models

- **Stay cached indefinitely** unless extreme memory pressure
- **Auto-marked as persistent** on first load
- **Protected from eviction** even when loading large models
- **Quick reuse** across different tools and agents

### Chat Models (Medium Size)

- **Stay cached for extended periods** (15-30 minutes default)
- **Evicted only when memory pressure exists**
- **Reused efficiently** for multiple conversations
- **Priority-based protection** for HIGH priority models

### Large Models (>10GB)

- **Cached but evictable** when new models need loading
- **Intelligent eviction order** (oldest/lowest priority first)
- **Memory pressure detection** prevents unnecessary clearing
- **Size penalties** but still get reasonable cache time

## Configuration

The system uses sensible defaults but can be tuned:

```python
# Cache timeout (base timeout before dynamic multipliers)
cache_manager = LocalPipelineCacheManager(cache_timeout=300)  # 5 minutes base

# Memory thresholds for categories (in pipeline_cache.py)
SMALL_MODEL_THRESHOLD = 2 * 1024**3     # 2GB
MEDIUM_MODEL_THRESHOLD = 5 * 1024**3    # 5GB  
LARGE_MODEL_THRESHOLD = 10 * 1024**3    # 10GB
```

## Testing

Run the test suite to validate the enhanced caching behavior:

```bash
# Simple eviction logic tests
kubectl exec -it -n ollama $POD_NAME -- /app/v.sh python -m debug.test_simple_enhanced_cache

# Full E2E validation  
kubectl exec -it -n ollama $POD_NAME -- /app/v.sh python -m debug.test_composer_real_e2e
```

## Architecture Integration

The enhanced caching system integrates cleanly with the existing architecture:

- **LocalPipelineCacheManager**: Core caching logic with intelligent eviction
- **PipelineFactory**: High-level interface with persistence control APIs
- **BaseAgent**: Modified cleanup to preserve cached pipelines
- **Hardware Manager**: Memory pressure detection and GPU management
- **Model Profiles**: Priority and configuration integration

## Benefits

### For Embedding Workflows

- ✅ **Instant responses** - no pipeline loading delays
- ✅ **Consistent performance** - embedding models always available
- ✅ **Resource efficiency** - small memory footprint with permanent caching

### For Multi-Model Workflows  

- ✅ **Smart resource management** - keeps valuable models while evicting when needed
- ✅ **Reduced loading times** - frequently used models stay cached
- ✅ **Better memory utilization** - size-aware decisions prevent waste

### For System Performance

- ✅ **Predictable behavior** - intelligent eviction vs random/aggressive clearing
- ✅ **Memory pressure handling** - responds to actual needs vs preemptive clearing
- ✅ **Monitoring and control** - visibility into cache status and manual controls

The system ensures that embedding pipelines (and other small, valuable models) remain in memory indefinitely, while larger models are managed intelligently based on memory pressure and usage patterns.
