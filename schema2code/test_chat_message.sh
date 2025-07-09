#!/bin/bash
# Test script for generating ChatMessage.proto with references to MessageType

# Define paths
SCHEMA_PATH="../schemas/chat_message.yaml"
OUTPUT_PATH="./chat_message.proto"

# Generate the Protocol Buffer file
echo "Generating Protocol Buffer file..."
python schema2code.py "$SCHEMA_PATH" --language proto --output "$OUTPUT_PATH" --package "proto" --go-package "proto"

# Show the result if successful
if [ $? -eq 0 ]; then
    echo "Successfully generated Protocol Buffer file at $OUTPUT_PATH"
    echo "Generated content:"
    echo "===================="
    cat "$OUTPUT_PATH"
    echo "===================="
else
    echo "Failed to generate Protocol Buffer file"
fi
