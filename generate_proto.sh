#!/bin/bash

# Script to generate both Go and Python code from protobuf definitions

set -e # Exit immediately if a command exits with a non-zero status

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=== Generating Proto Code for Python and Go ==="
echo "Working directory: $(pwd)"

# Check for required dependencies
echo "Checking dependencies..."

# Check for protoc
if ! command -v protoc &>/dev/null; then
    echo "Error: protoc is not installed. Please install Protocol Buffers compiler."
    exit 1
fi
echo "✅ protoc found: $(which protoc)"

# Check for Python dependencies
if ! python -c "import grpc_tools.protoc" &>/dev/null; then
    echo "Error: grpc_tools.protoc not found. Please install with: pip install grpcio-tools"
    exit 1
fi
echo "✅ grpc_tools.protoc found"

# Check if required Go tools are installed
if ! command -v protoc-gen-go &>/dev/null; then
    echo "Warning: protoc-gen-go not found. It will be installed by the Go proto generator script."
fi

if ! command -v protoc-gen-go-grpc &>/dev/null; then
    echo "Warning: protoc-gen-go-grpc not found. It will be installed by the Go proto generator script."
fi

# Check if proto directory and file exist
if [ ! -d "proto" ]; then
    echo "Error: proto directory not found! Creating it..."
    mkdir -p proto
fi

if [ ! -f "proto/inference.proto" ]; then
    echo "Error: proto/inference.proto not found!"
    echo "Current directory: $(pwd)"
    echo "Available files in current directory:"
    ls -la
    echo "Available files in proto directory (if exists):"
    if [ -d "proto" ]; then
        ls -la proto
    fi
    exit 1
fi

echo "Proto file exists: ${SCRIPT_DIR}/proto/inference.proto"

echo ""
echo "1. Generating Python Proto Code..."
cd inference/grpc_server/proto
python generate_proto.py
if [ $? -ne 0 ]; then
    echo "Error generating Python proto code"
    exit 1
fi
cd "${SCRIPT_DIR}"

echo ""
echo "2. Generating Go Proto Code..."
cd maistro/grpc
bash generate_proto.sh
if [ $? -ne 0 ]; then
    echo "Error generating Go proto code"
    exit 1
fi
cd "${SCRIPT_DIR}"

echo ""
echo "Proto code generation completed successfully for both languages!"
echo "Python code: ${SCRIPT_DIR}/inference/grpc_server/proto/"
echo "Go code: ${SCRIPT_DIR}/maistro/grpc/inference/"
