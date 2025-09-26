#!/bin/bash

# Search Provider Validation Test Wrapper
set -e

echo "🚀 Running Search Provider Validation Tests..."
echo "Activating runner environment..."

# Activate the runner environment
source /app/v.sh runner

echo "Virtual environment: $VIRTUAL_ENV"
echo "Working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"

# Run the search provider validation test
cd /app
python test_search_providers_validation.py