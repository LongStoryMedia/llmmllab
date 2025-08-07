#!/bin/bash
# Script to split a large GGUF file into multiple shards
# Usage: ./split_gguf_model.sh <input_gguf_file> <output_dir> <num_shards>

set -e

INPUT_GGUF="$1"
OUTPUT_DIR="$2"
NUM_SHARDS="$3"

if [ -z "$INPUT_GGUF" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$NUM_SHARDS" ]; then
    echo "Usage: ./split_gguf_model.sh <input_gguf_file> <output_dir> <num_shards>"
    echo "Example: ./split_gguf_model.sh /models/qwen2.5-vl-72b-instruct.gguf /models/qwen2.5-split 2"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# --split: split GGUF to multiple GGUF, default operation.
# --split-max-size: max size per split in M or G, f.ex. 500M or 2G.
# --split-max-tensors: maximum tensors in each split: default(128)
# --merge: merge multiple GGUF to a single GGUF.
# Run the convert tool to create model shards
echo "Splitting model into $NUM_SHARDS shards..."
/llama.cpp/build/bin/llama-gguf-split 

echo "Model has been split into $NUM_SHARDS shards in $OUTPUT_DIR"
echo "You can load these shards using the split model path pattern:"
echo "  $OUTPUT_DIR/qwen2.5-vl-split.*.gguf"
