#!/bin/bash

# Real-World Search Scenarios Test Wrapper for Kubernetes
set -e

echo "🚀 Running Real-World Search Scenarios Test..."
echo "Activating runner environment..."

# Activate the runner environment
source /app/v.sh runner

echo "Virtual environment: $VIRTUAL_ENV"
echo "Working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"

# Run the real search scenarios test
cd /app
python test_real_search_scenarios.py