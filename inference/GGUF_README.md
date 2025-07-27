# GGUF Model Pipeline with llama-cpp-python

This implementation uses llama-cpp-python to run quantized GGUF models efficiently with CUDA acceleration.

## Setup Instructions

### 1. Install llama-cpp-python with CUDA support

Use the provided installation script:

```bash
chmod +x install_llama_cpp.sh
./install_llama_cpp.sh
```

Or install manually with:

```bash
# Set environment variables for CUDA compilation
export CUDA_HOME=/usr/local/cuda
export LLAMA_CUBLAS=1
export CMAKE_ARGS="-DLLAMA_CUBLAS=on"
export FORCE_CMAKE=1

# Install llama-cpp-python with CUDA support
pip install llama-cpp-python
pip install transformers
```

### 2. Verify GPU Support

```python
from llama_cpp import Llama
import torch

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# Test llama-cpp-python GPU support
model = Llama(
    model_path="path/to/your/model.gguf",
    n_gpu_layers=-1,  # -1 means all layers on GPU
    verbose=True
)
```

### 3. Model Configuration

In `models.json`, ensure your GGUF model entry has:
- Correct `pipeline` field set to `Qwen30A3BQ4KMPipe` 
- `gguf_file` property in `details` pointing to the model path
- `parent_model` in `details` for the tokenizer

Example:
```json
{
    "id": "qwen2-30b-a3b-q4-k-m",
    "name": "unsloth/Qwen3-30B-A3B-GGUF",
    "model": "/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf",
    "pipeline": "Qwen30A3BQ4KMPipe",
    "details": {
        "parent_model": "Qwen/Qwen3-30B-A3B-Instruct",
        "gguf_file": "/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf",
        "format": "gguf",
        "quantization_level": "Q4_K_M"
    }
}
```

### 4. Running the Test

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run the test script
python inference/test_qwen_gguf_pipeline.py
```

## Performance Optimization

### GPU Memory Optimization

- Adjust `n_gpu_layers` based on your GPU memory. For RTX 3090/3060:
  - RTX 3090 (24GB): Can handle most models with `n_gpu_layers=-1`
  - RTX 3060 (12GB): May need to use `n_gpu_layers=20` or lower

### Batch Size Tuning

- Adjust `n_batch` parameter in Llama initialization:
  ```python
  model = Llama(
      model_path="path/to/model.gguf",
      n_batch=512  # Try different values: 128, 256, 512, 1024
  )
  ```

### Context Length

- For longer contexts, increase `n_ctx` but be aware of increased memory usage:
  ```python
  model = Llama(
      model_path="path/to/model.gguf",
      n_ctx=8192  # Default is 4096, can go up to 32k depending on model
  )
  ```

## Troubleshooting

### Common Issues

1. **CUDA Error: Out of memory**
   - Reduce `n_gpu_layers` or `n_ctx`
   - Try a more aggressively quantized model (e.g., Q4_0 instead of Q5_K_M)

2. **Slow Inference**
   - Increase `n_batch` parameter
   - Ensure CUDA is being used properly

3. **Tokenizer Mismatch**
   - Ensure `parent_model` points to the correct HuggingFace model for tokenization

4. **Missing Thinking Mode**
   - Verify chat template is correctly applying the thinking mode tokens

### Debugging

Enable verbose mode to see detailed loading and inference information:

```python
model = Llama(
    model_path="path/to/model.gguf",
    verbose=True
)
```
