#!/bin/bash

# End-to-End Tool Calling Test Wrapper
set -e

echo "🚀 Running End-to-End Tool Calling Test..."
echo "Activating runner environment..."

# Activate the runner environment
source /app/v.sh runner

echo "Virtual environment: $VIRTUAL_ENV"
echo "Working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"

# Run the end-to-end tool calling test
cd /app
python test_end_to_end_tool_calling.py