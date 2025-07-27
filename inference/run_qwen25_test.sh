#!/bin/bash
# Script to run the Qwen 2.5 VL GGUF pipeline test with optimized memory settings

# Set the directory to the script's location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || exit

# Source the configuration file
echo "Loading optimized configuration settings..."
source ./qwen25_vl_gguf_config.sh

# Run the test
echo "Starting test with memory-optimized settings..."
python -m manual_tests.test_qwen25_vl_gguf_pipeline
