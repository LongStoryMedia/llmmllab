# Parameter Optimization Control Guide

## Summary of Issues Fixed

The parameter optimization system had several critical bugs that have been resolved:

### 🐛 **Major Bugs Fixed:**
1. **Recovery strategy broken**: `move_to_cpu` strategy wasn't actually reducing `n_gpu_layers`
2. **Context size waste**: Using 65536 context when model trained on 40960 wasted ~2GB memory
3. **Async warnings**: Memory preallocation tests weren't properly awaited
4. **Poor layer movement**: No visibility into actual GPU layer reduction

### ⚡ **Performance Improvements:**
- **Context size now limited to training context** (40960 instead of 65536 for Qwen3-4B)
- **Recovery strategy now actually moves layers to CPU** with proper logging
- **Memory estimates reduced** from 10GB to ~7-8GB for realistic configurations
- **Better debugging** with layer movement visibility

## How to Disable Parameter Optimization

If you want to disable the "back and forth" parameter optimization entirely:

### Option 1: Environment Variable (Recommended)
```bash
# Set this environment variable to disable optimization
export PARAMETER_OPTIMIZATION_ENABLED=false
```

### Option 2: User Config File
Edit your user config file to include:
```yaml
parameter_optimization:
  enabled: false
```

### Option 3: Model Profile Override
In your model profile configuration, you can specify exact parameters:
```yaml
parameters:
  num_ctx: 8192      # Use smaller context
  batch_size: 256    # Use smaller batch
gpu_config:
  gpu_layers: 20     # Use fewer GPU layers
```

## Expected Behavior After Fixes

With the fixes applied, you should see:

1. **Faster recovery**: Actual layer reduction instead of repeated failures
2. **Better memory usage**: Context limited to model's training size
3. **Clear logging**: You'll see messages like "Moving 5 layers to CPU: 32 → 27"
4. **Fewer attempts**: Recovery should succeed sooner with proper parameter reduction

## Test the Fixes

Try loading a model again. You should see:
- Context automatically limited to training size (40960 for Qwen3-4B)
- Actual GPU layer reduction in recovery attempts
- Faster successful initialization

The system will now work much more efficiently without the "waaay up, then waaay back down" behavior you experienced.