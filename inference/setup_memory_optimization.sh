#!/bin/bash

# Enhanced memory and CUDA management verification script for K8s
# This script verifies that environment variables are properly set in the container

echo "=========================================="
echo "CUDA and Memory Optimization Status Check"
echo "=========================================="

# Function to check environment variable
check_env() {
    local var_name=$1
    local expected_value=$2
    local actual_value=$(printenv $var_name)
    
    if [ "$actual_value" = "$expected_value" ]; then
        echo "✓ $var_name = $actual_value"
    else
        echo "✗ $var_name = $actual_value (expected: $expected_value)"
    fi
}

echo ""
echo "CUDA Graphs Disabling:"
check_env "LLAMA_CUDA_USE_GRAPHS" "0"
check_env "GGML_CUDA_USE_GRAPHS" "0"
check_env "CUDA_USE_GRAPHS" "0"
check_env "LLAMA_GRAPH" "0"
check_env "GGML_GRAPH" "0"

echo ""
echo "CUBLAS Configuration:"
check_env "LLAMA_CUBLAS" "1"
check_env "GGML_CUDA_FORCE_CUBLAS" "1"
check_env "GGML_CUDA_FORCE_MMQ" "0"
check_env "GGML_USE_CUBLAS" "1"

echo ""
echo "Memory Management:"
check_env "GGML_CUDA_POOL_SIZE" "2147483648"
check_env "LLAMA_CUDA_POOL_SIZE" "2147483648"
check_env "CUDA_MEMORY_FRACTION" "0.8"
check_env "GGML_CUDA_NO_PINNED" "1"

echo ""
echo "Device Configuration:"
check_env "CUDA_VISIBLE_DEVICES" "0,1,2"
check_env "CUDA_DEVICE_ORDER" "PCI_BUS_ID"

echo ""
echo "Python Memory Optimization:"
check_env "PYTHONMALLOC" "malloc"
check_env "MALLOC_ARENA_MAX" "2"

echo ""
echo "Logging:"
check_env "GGML_LOG_LEVEL" "2"

echo ""
echo "=========================================="
echo "CUDA Device Information:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
else
    echo "nvidia-smi not available"
fi

echo ""
echo "GPU Memory Usage by Process:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi pmon -c 1 2>/dev/null || echo "No processes found"
else
    echo "nvidia-smi not available"
fi

echo ""
echo "=========================================="
echo "Verification complete!"
echo ""
echo "To run this check in K8s:"
echo "kubectl exec -n ollama <pod-name> -- /app/setup_memory_optimization.sh"
