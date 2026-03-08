#!/bin/bash
# setup_cuda_runtime.sh - Ensure proper CUDA runtime environment

set -e

echo "Setting up CUDA runtime environment..."

# Remove CUDA stubs from LD_LIBRARY_PATH if they exist
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed 's|/usr/local/cuda/lib64/stubs:||g' | sed 's|:/usr/local/cuda/lib64/stubs||g' | sed 's|^/usr/local/cuda/lib64/stubs$||g')

# Ensure proper CUDA runtime libraries are in the path
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/llama.cpp/bin/build:${LD_LIBRARY_PATH}"

# Set CUDA device environment
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2"}
export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-"all"}
export NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-"compute,utility,video"}

# Additional CUDA environment variables
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_CACHE_PATH="/tmp/cuda_cache"
export CUDA_LAUNCH_BLOCKING=0

# Create CUDA cache directory
mkdir -p "$CUDA_CACHE_PATH"

# Log CUDA environment for debugging
echo "CUDA Environment:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  NVIDIA_VISIBLE_DEVICES: $NVIDIA_VISIBLE_DEVICES"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "  CUDA_DEVICE_ORDER: $CUDA_DEVICE_ORDER"

# Check if CUDA runtime libraries are accessible
if ldconfig -p | grep -q "libcuda.so"; then
    echo "✓ CUDA runtime library found in ldconfig cache"
else
    echo "⚠ CUDA runtime library not found in ldconfig cache"
    echo "Available CUDA libraries:"
    ldconfig -p | grep cuda || echo "No CUDA libraries found"
fi

# Check for nvidia-smi availability
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "✓ nvidia-smi is available"
    nvidia-smi --list-gpus || echo "⚠ Could not list GPUs"
else
    echo "⚠ nvidia-smi not available"
fi

echo "CUDA runtime setup complete."