#!/bin/bash
# Environment variables for optimizing Qwen 2.5 VL GGUF model loading
# Source this file before running the test script

# The number of layers to load onto GPU (out of 80 total layers in the model)
# Setting this lower will use less GPU memory but potentially reduce performance
export QWEN25_VL_GGUF_GPU_LAYERS=35

# The context size for the model (in tokens)
# Smaller context size uses less memory but may affect long conversations
export QWEN25_VL_GGUF_CTX_SIZE=8192

# Main GPU to use (0 = first GPU)
export QWEN25_VL_GGUF_MAIN_GPU=0

# Log these settings
echo "Qwen 2.5 VL GGUF Configuration:"
echo "GPU Layers: $QWEN25_VL_GGUF_GPU_LAYERS"
echo "Context Size: $QWEN25_VL_GGUF_CTX_SIZE"
echo "Main GPU: $QWEN25_VL_GGUF_MAIN_GPU"
