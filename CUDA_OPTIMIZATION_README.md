# CUDA Optimization and Context Window Fixes

## Overview

This document describes the comprehensive fixes applied to resolve CUDA memory issues and context window limitations in the LLM inference system.

## Issues Addressed

1. **CUDA Out of Memory**: "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1468.00 MiB on device 1: cudaMalloc failed: out of memory"
2. **Context Window Exceeded**: "Requested tokens (4267/4722) exceed context window of 2048"
3. **CUDA Graphs Enabled**: Despite environment variables, logs showed "USE_GRAPHS = 1"
4. **Degraded Model Output**: Responses reduced to single words like "NO"

## Fixes Applied

### 1. Enhanced Context Management

**File**: `inference/runner/pipelines/txt2txt/context_manager.py`
- New context manager class for intelligent message truncation
- Token estimation and context window management
- Automatic recovery from context overflow errors
- Preserves system messages and most recent user input

**File**: `inference/runner/pipelines/txt2txt/qwen3moe.py`
- Increased context window from 2048 to 8192+ tokens for 30B models
- Integrated context manager for proactive message truncation
- Enhanced error handling with automatic context recovery
- Dynamic context sizing based on available VRAM

### 2. Comprehensive CUDA Optimization

**Files**: 
- `inference/k8s/deployment.yaml`
- `inference/k8s/env.yaml`

Added extensive CUDA environment variables:

```yaml
# CUDA Graphs Disabling (multiple variables for different builds)
LLAMA_CUDA_USE_GRAPHS: "0"
GGML_CUDA_USE_GRAPHS: "0" 
CUDA_USE_GRAPHS: "0"
LLAMA_GRAPH: "0"
GGML_GRAPH: "0"

# Force Basic CUBLAS (avoid optimized routines that cause issues)
LLAMA_CUBLAS: "1"
GGML_CUDA_FORCE_CUBLAS: "1"
GGML_CUDA_FORCE_MMQ: "0"
GGML_USE_CUBLAS: "1"

# Memory Pool Limits (2GB per GPU for 3-GPU setup)
GGML_CUDA_POOL_SIZE: "2147483648"
LLAMA_CUDA_POOL_SIZE: "2147483648"
CUDA_MEMORY_FRACTION: "0.8"

# Additional Optimizations
GGML_CUDA_NO_PINNED: "1"
CUDA_VISIBLE_DEVICES: "0,1,2"
CUDA_DEVICE_ORDER: "PCI_BUS_ID"
PYTHONMALLOC: "malloc"
MALLOC_ARENA_MAX: "2"
```

### 3. Pipeline Optimizations

**Qwen Pipeline Changes**:
- Reduced batch size from 128 to 64 for memory stability
- Dynamic context sizing based on model size and available VRAM
- Conservative memory management with `use_mmap=True`, `use_mlock=False`
- Enhanced error recovery with context truncation fallback

## Kubernetes Deployment

### Environment Variables Applied To:

1. **Main Ollama Deployment** (`deployment.yaml`)
   - Namespace: `ollama`
   - Pod: `ollama`
   - Resources: 3 GPUs, 24-30Gi memory
   - All CUDA optimization variables added directly

2. **Inference Service Deployment** (`inference-service.yaml`)
   - Namespace: `ollama` 
   - Pod: `inference-service`
   - Resources: 1 GPU, 4-6Gi memory
   - Uses ConfigMap (`inference-config`) with all optimization variables

### Verification

Use the verification script to check environment variables:

```bash
# Copy verification script to pod
kubectl cp setup_memory_optimization.sh ollama/<pod-name>:/app/

# Run verification
kubectl exec -n ollama <pod-name> -- /app/setup_memory_optimization.sh
```

### Monitoring

Monitor CUDA memory usage:
```bash
kubectl exec -n ollama <pod-name> -- nvidia-smi
```

Check logs for CUDA optimization status:
```bash
kubectl logs -n ollama <pod-name> | grep -E "(USE_GRAPHS|CUDA|memory)"
```

## Expected Results

1. **No More CUDA OOM**: Memory pool limits prevent excessive allocation
2. **Proper Context Handling**: 8192+ token context with automatic truncation
3. **Quality Responses**: Full model responses instead of truncated output
4. **Stable Operation**: No more pipeline reloads due to memory pressure

## Testing

1. **Load a conversation with 20+ messages** - should handle context gracefully
2. **Run concurrent requests** - memory pools should prevent conflicts
3. **Monitor GPU memory** - should stay within configured limits
4. **Check model output quality** - should provide full responses

## Rollback Plan

If issues occur, revert environment variables by:

1. Remove CUDA optimization variables from `deployment.yaml`
2. Remove CUDA section from `env.yaml` 
3. Restart deployments: `kubectl rollout restart deployment -n ollama`

## Notes

- Changes are applied at the Kubernetes ConfigMap/Deployment level
- Environment variables take effect on pod restart
- The context manager provides graceful degradation for long conversations
- Memory limits are conservative to ensure stability across different workloads
