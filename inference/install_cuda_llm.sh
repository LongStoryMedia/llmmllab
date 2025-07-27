#!/bin/bash
# Script to install llama-cpp-python with CUDA support

set -e  # Exit immediately if a command exits with a non-zero status

echo "Installing llama-cpp-python with CUDA support..."

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "WARNING: CUDA toolkit not found, trying to install anyway"
fi

# Uninstall existing llama-cpp-python
pip uninstall -y llama-cpp-python

# Install with CUDA support
export CMAKE_ARGS="-DLLAMA_CUBLAS=on"
pip install --no-cache-dir llama-cpp-python

# Verify installation
echo "Verifying CUDA support in llama-cpp-python..."
python3 -c "
import torch
from llama_cpp import Llama

print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA devices:', torch.cuda.device_count())
    print('CUDA device name:', torch.cuda.get_device_name(0))

# Check if llama-cpp-python was compiled with CUDA
print('Checking llama-cpp-python CUDA support...')
try:
    # Create a small model with GPU layers to test CUDA support
    # This will only work if CUDA support is enabled
    model = Llama(
        model_path=None,  # No model loaded, just checking CUDA support
        n_gpu_layers=1,
        seed=0
    )
    print('llama-cpp-python has CUDA support')
except Exception as e:
    if 'n_gpu_layers > 0 but GGML_USE_CUBLAS is not defined' in str(e):
        print('ERROR: llama-cpp-python was not compiled with CUDA support')
    else:
        print('Error checking CUDA support:', str(e))
"

echo "Installation complete"
