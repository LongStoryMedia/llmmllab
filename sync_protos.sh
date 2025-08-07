#!/bin/bash

# This script ensures both Go and Python services use identical proto definitions

set -e

# Define directories
PROTO_DIR="./proto"
GO_OUT_DIR="./maistro/proto"
PYTHON_OUT_DIR="./inference/protos"

echo "Syncing proto files between Go and Python services"

# Step 1: Compile protos with the same version of protoc
echo "Compiling proto files..."
protoc \
    --proto_path=${PROTO_DIR} \
    --go_out=${GO_OUT_DIR} \
    --go_opt=paths=source_relative \
    --go-grpc_out=${GO_OUT_DIR} \
    --go-grpc_opt=paths=source_relative \
    --python_out=${PYTHON_OUT_DIR} \
    --grpc_python_out=${PYTHON_OUT_DIR} \
    ${PROTO_DIR}/*.proto

echo "✓ Proto files compiled for both Go and Python"

# Step 2: Create proper Python __init__.py for imports
cat >${PYTHON_OUT_DIR}/__init__.py <<EOF
"""
Protocol buffer module initialization.
"""

# Import all modules to make them available
from os.path import dirname, basename, isfile, join
import glob

__all__ = []

# Add all proto modules to __all__
for f in glob.glob(join(dirname(__file__), "*_pb2*.py")):
    if isfile(f) and not f.endswith('__init__.py'):
        module = basename(f)[:-3]
        __all__.append(module)
        # Import the module to make it available
        exec(f"from . import {module}")
EOF

echo "✓ Created Python package __init__.py"

# Step 3: Test importing the modules
echo "Testing Python imports..."
python -c "from inference.protos import inference_pb2, inference_pb2_grpc; print('Python imports successful')"
if [ $? -eq 0 ]; then
    echo "✓ Python proto imports verified"
else
    echo "✗ Python proto imports failed"
    exit 1
fi

echo "Proto synchronization complete"
