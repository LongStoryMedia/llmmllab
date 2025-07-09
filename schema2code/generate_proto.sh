#!/bin/bash
# Generate all protobuf files in the correct order

set -e # Exit on error

SCHEMA_DIR="../schemas"
OUTPUT_DIR="../proto"
PACKAGE_NAME="proto"
GO_PACKAGE="github.com/llmmllab/proto"

echo "Generating Protocol Buffer files..."
echo "Step 1: Generate message_type.proto"
python schema2code.py "${SCHEMA_DIR}/message_type.yaml" --language "proto" -o "${OUTPUT_DIR}/message_type.proto" --go-package "${GO_PACKAGE}" --package "${PACKAGE_NAME}"

echo "Step 2: Generate chat_message.proto"
python schema2code.py "${SCHEMA_DIR}/chat_message.yaml" --language "proto" -o "${OUTPUT_DIR}/chat_message.proto" --go-package "${GO_PACKAGE}" --package "${PACKAGE_NAME}"

echo "Step 3: Generate chat_req.proto"
python schema2code.py "${SCHEMA_DIR}/chat_req.yaml" --language "proto" -o "${OUTPUT_DIR}/chat_req.proto" --go-package "${GO_PACKAGE}" --package "${PACKAGE_NAME}"

echo "======================================="
echo "Proto file generation completed successfully!"

# Verify the imports in each file
echo "Verifying file imports..."

echo "chat_message.proto imports:"
grep "^import" "${OUTPUT_DIR}/chat_message.proto" || echo "No imports found"

echo "chat_req.proto imports:"
grep "^import" "${OUTPUT_DIR}/chat_req.proto" || echo "No imports found"

echo "Done!"

echo "Generation completed successfully!"
echo "Files generated:"
ls -la "${OUTPUT_DIR}"/*.proto
