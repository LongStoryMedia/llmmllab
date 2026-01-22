# GPU Configuration Guide

This document explains how to use the GPU configuration system in LLM ML Lab to optimize performance and memory usage across multiple GPUs.

## Overview

The GPU configuration system provides fine-grained control over:
- **Device Selection**: Choose which GPU(s) to use for model inference
- **Memory Management**: Control KV cache offloading and memory allocation
- **Tensor Splitting**: Distribute model layers across multiple GPUs
- **MoE Optimization**: Configure CPU layer allocation for Mixture-of-Experts models

## Configuration Structure

GPU configuration is specified in the `gpu_config` field of model parameters:

```yaml
# Example model profile configuration
parameters:
  num_ctx: 4096
  temperature: 0.7
  gpu_config:
    no_kv_offload: false
    main_gpu: 1
    main_gpu_device_id: "NVIDIA GeForce RTX 3090"
    tensor_split: [0.4, 0.6]
    tensor_split_devices: ["GPU 0", "GPU 1"]
    n_cpu_moe: 2
    split_mode: "layer"
    offload_kqv: true
```

## Configuration Parameters

### Device Selection

#### `main_gpu` (integer, default: -1)
Specifies the primary GPU device index for model loading.
- `-1`: Auto-selection (default)
- `0, 1, 2, ...`: Specific GPU index

**Example:**
```yaml
gpu_config:
  main_gpu: 1  # Use GPU 1 as primary device
```

#### `main_gpu_device_id` (string, optional)
Alternative to `main_gpu` using user-friendly device names. Overrides `main_gpu` if specified.

**Example:**
```yaml
gpu_config:
  main_gpu_device_id: "NVIDIA GeForce RTX 3090"
```

### Memory Management

#### `no_kv_offload` (boolean, default: false)
Forces KV cache to stay on CPU instead of GPU, saving VRAM at the cost of performance.

**Example:**
```yaml
gpu_config:
  no_kv_offload: true  # Keep KV cache on CPU to save VRAM
```

#### `offload_kqv` (boolean, default: true)
Controls whether key/query/value tensors are offloaded to GPU. Opposite of `no_kv_offload`.

**Example:**
```yaml
gpu_config:
  offload_kqv: false  # Equivalent to no_kv_offload: true
```

### Multi-GPU Configuration

#### `tensor_split` (array of floats)
Defines how to split the model across multiple GPUs. Values must sum to 1.0.

**Example:**
```yaml
gpu_config:
  tensor_split: [0.3, 0.7]  # 30% on GPU 0, 70% on GPU 1
```

#### `tensor_split_devices` (array of strings, optional)
Specifies device names/IDs corresponding to `tensor_split` fractions.

**Example:**
```yaml
gpu_config:
  tensor_split: [0.4, 0.6]
  tensor_split_devices: 
    - "NVIDIA GeForce RTX 2060"
    - "NVIDIA GeForce RTX 3090"
```

### MoE Model Optimization

#### `n_cpu_moe` (integer, default: 0)
Number of Mixture-of-Experts layers to keep on CPU. Useful for large MoE models that don't fit entirely in GPU memory.

**Example:**
```yaml
gpu_config:
  n_cpu_moe: 2  # Keep 2 MoE layers on CPU
```

#### `split_mode` (string, default: "layer")
Determines how to split the model across devices.

**Example:**
```yaml
gpu_config:
  split_mode: "layer"  # Split by layers (default)
```

## Usage Examples

### Single GPU Optimization

For models that fit on a single GPU but need memory optimization:

```yaml
parameters:
  gpu_config:
    main_gpu: 0
    no_kv_offload: false  # Use GPU for KV cache
    offload_kqv: true     # Offload tensors to GPU
```

### Multi-GPU Load Balancing

For large models that need to be split across multiple GPUs:

```yaml
parameters:
  gpu_config:
    tensor_split: [0.4, 0.6]
    tensor_split_devices:
      - "NVIDIA GeForce RTX 2060"  # 40% of model
      - "NVIDIA GeForce RTX 3090"  # 60% of model
```

### Memory-Constrained Setup

For systems with limited VRAM:

```yaml
parameters:
  gpu_config:
    main_gpu: 0
    no_kv_offload: true   # Keep KV cache on CPU
    n_cpu_moe: 3          # Keep 3 MoE layers on CPU
```

### High-Performance Setup

For systems with ample GPU memory:

```yaml
parameters:
  gpu_config:
    main_gpu_device_id: "NVIDIA GeForce RTX 3090"
    offload_kqv: true
    no_kv_offload: false
```

### MoE Model Configuration

For Mixture-of-Experts models like Qwen3-MoE:

```yaml
parameters:
  gpu_config:
    main_gpu: 1
    n_cpu_moe: 2          # Keep some layers on CPU
    tensor_split: [0.5, 0.5]  # Split remaining across 2 GPUs
```

## API Integration

### Device Discovery

Get available devices and their names:

```bash
curl -X GET "http://localhost:8000/v1/resources/devices"
```

Response:
```json
{
  "devices": {
    "0": {
      "index": 0,
      "name": "NVIDIA GeForce RTX 2060",
      "uuid": "GPU-22fff4f7-d40b-1ffd-48c7-43c48821ea9c",
      "id": "0"
    },
    "1": {
      "index": 1,
      "name": "NVIDIA GeForce RTX 3090",
      "uuid": "GPU-6a792ee8-fbcb-869f-dbe0-05431e3d3355",
      "id": "1"
    },
    "cpu": {
      "index": -1,
      "name": "CPU",
      "uuid": "",
      "id": "cpu"
    }
  }
}
```

### Device Name Resolution

Resolve device names to indices:

```bash
curl -X POST "http://localhost:8000/v1/resources/devices/resolve" \
  -H "Content-Type: application/json" \
  -d '{"device_name_or_index": "NVIDIA GeForce RTX 3090"}'
```

Response:
```json
{
  "resolved_index": 1,
  "device_name": "NVIDIA GeForce RTX 3090"
}
```

## Best Practices

### 1. Device Selection
- Use device names (`main_gpu_device_id`) instead of indices for better portability
- Check available devices via API before configuring

### 2. Memory Management
- Start with `no_kv_offload: false` for best performance
- Enable `no_kv_offload: true` only if you encounter VRAM issues
- Monitor GPU memory usage to find optimal settings

### 3. Multi-GPU Setup
- Ensure `tensor_split` values sum to 1.0
- Place larger portions on GPUs with more VRAM
- Use device names in `tensor_split_devices` for clarity

### 4. MoE Models
- Start with `n_cpu_moe: 0` and increase if memory issues occur
- Monitor inference speed vs memory usage when adjusting CPU layers

### 5. Performance Tuning
- Test different configurations with your specific models
- Use monitoring endpoints to track GPU utilization
- Adjust based on actual workload patterns

## Troubleshooting

### Common Issues

1. **"Device not found" errors**
   - Check device names with `/v1/resources/devices` endpoint
   - Verify GPU drivers and CUDA installation

2. **Memory allocation failures**
   - Reduce model size with `tensor_split`
   - Enable `no_kv_offload: true`
   - Increase `n_cpu_moe` for MoE models

3. **Poor performance**
   - Ensure proper GPU utilization with monitoring
   - Avoid unnecessary CPU offloading
   - Balance tensor split based on GPU capabilities

### Monitoring

Use the health endpoint to monitor GPU status:

```bash
curl -X GET "http://localhost:8000/v1/resources/health"
```

This returns GPU temperature, memory utilization, and process information.

## Schema Reference

The complete GPU configuration schema:

```yaml
gpu_config:
  type: object
  properties:
    no_kv_offload:
      type: boolean
      default: false
      description: "Force KV cache to CPU instead of GPU (saves VRAM)"
    
    main_gpu:
      type: integer
      default: -1
      minimum: -1
      description: "Main GPU device index (-1 for auto-selection)"
    
    main_gpu_device_id:
      type: string
      description: "Device ID/name for main GPU (overrides main_gpu index)"
    
    tensor_split:
      type: array
      items:
        type: number
      description: "Fraction of model to put on each GPU (must sum to 1.0)"
    
    tensor_split_devices:
      type: array
      items:
        type: string
      description: "Device IDs/names corresponding to tensor_split fractions"
    
    n_cpu_moe:
      type: integer
      default: 0
      minimum: 0
      description: "Number of MoE layers to keep on CPU (gpt-oss models only)"
    
    split_mode:
      type: string
      default: "layer"
      description: "How to split model across devices"
    
    offload_kqv:
      type: boolean
      default: true
      description: "Offload key/query/value tensors to GPU"
```

This configuration system provides comprehensive control over GPU resource allocation while maintaining ease of use for common scenarios.