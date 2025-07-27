#!/bin/bash

# Script to extract GGUF and mmproj files from Ollama's downloaded models
# Run this after: ollama pull qwen2.5-vl:32b

echo "Extracting Qwen2.5-VL files from Ollama..."

# Find Ollama models directory
if [[ "$OSTYPE" == "darwin"* ]]; then
    OLLAMA_DIR="$HOME/.ollama"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OLLAMA_DIR="$HOME/.ollama"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OLLAMA_DIR="$USERPROFILE/.ollama"
else
    echo "Unsupported OS"
    exit 1
fi

BLOBS_DIR="$OLLAMA_DIR/models/blobs"
OUTPUT_DIR="/models/qwen2.5-vl-32b-instruct"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Looking for files in: $BLOBS_DIR"

# Look for the model files (they will be the largest files)
echo "Finding GGUF model file (largest file, likely 15GB+)..."
GGUF_FILE=$(find "$BLOBS_DIR" -type f -size +10G | head -1)

if [ -n "$GGUF_FILE" ]; then
    echo "Found GGUF file: $GGUF_FILE"
    cp "$GGUF_FILE" "$OUTPUT_DIR/qwen2.5-vl-32b-instruct.gguf"
    echo "Copied to: $OUTPUT_DIR/qwen2.5-vl-32b-instruct.gguf"
else
    echo "No large GGUF file found. Model might not be downloaded."
fi

# Look for mmproj file (smaller, typically 100-500MB)
echo "Finding mmproj file (smaller file, typically 100-500MB)..."
MMPROJ_FILE=$(find "$BLOBS_DIR" -type f -size +50M -size -1G | head -1)

if [ -n "$MMPROJ_FILE" ]; then
    echo "Found mmproj file: $MMPROJ_FILE"
    cp "$MMPROJ_FILE" "$OUTPUT_DIR/mmproj.gguf"
    echo "Copied to: $OUTPUT_DIR/mmproj.gguf"
else
    echo "No mmproj file found."
fi

# List all files with sizes for manual identification if needed
echo ""
echo "All files in blobs directory (sorted by size):"
ls -lhS "$BLOBS_DIR"

echo ""
echo "Files extracted to: $OUTPUT_DIR"
echo "Update your model configuration to use these paths:"
echo "  GGUF file: $OUTPUT_DIR/qwen2.5-vl-32b-instruct.gguf"
echo "  mmproj file: $OUTPUT_DIR/mmproj.gguf"