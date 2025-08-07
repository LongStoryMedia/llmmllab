#!/bin/bash

# Standalone script to generate Go code from protobuf definitions
# This script should be run from the project root directory

set -e # Exit immediately if a command exits with a non-zero status

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Project directories
PROJECT_ROOT="${SCRIPT_DIR}"
PROTO_DIR="${PROJECT_ROOT}/proto"
OUTPUT_DIR="${PROJECT_ROOT}/maistro/proto"

echo "=== Generating Go Proto Code ==="
echo "Working directory: $(pwd)"
echo "Project root: ${PROJECT_ROOT}"
echo "Proto directory: ${PROTO_DIR}"
echo "Output directory: ${OUTPUT_DIR}"

# Check if proto file exists
if [ ! -f "${PROTO_DIR}/inference.proto" ]; then
    echo "Error: ${PROTO_DIR}/inference.proto not found!"
    exit 1
fi

# Check if protoc is installed
if ! command -v protoc &>/dev/null; then
    echo "Error: protoc is not installed. Please install Protocol Buffers compiler."
    exit 1
fi

# Check if Go is installed
if ! command -v go &>/dev/null; then
    echo "Error: go is not installed. Please install Go."
    exit 1
fi

# Install required Go packages if not already installed
echo "Installing required Go packages..."
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Get Go paths
GO_PATH=$(go env GOPATH)
echo "GOPATH: ${GO_PATH}"

# Ensure protoc-gen-go and protoc-gen-go-grpc are in the PATH
export PATH="${GO_PATH}/bin:${PATH}"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}/inference"

echo "Generating Go code from proto files..."
protoc \
    --proto_path="${PROTO_DIR}" \
    --go_out="${OUTPUT_DIR}" \
    --go_opt=paths=source_relative \
    --go-grpc_out="${OUTPUT_DIR}" \
    --go-grpc_opt=paths=source_relative \
    $(find "${PROTO_DIR}" -name "*.proto" -print0 | xargs -0)

# Check if generation was successful
if [ $? -eq 0 ]; then
    echo "Go code generation completed successfully!"
    echo "Generated files:"
    ls -la "${OUTPUT_DIR}/inference"*.go
else
    echo "Error: Failed to generate Go code"
    exit 1
fi
