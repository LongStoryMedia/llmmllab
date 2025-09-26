#!/bin/bash

# Comprehensive Search Validation Test Wrapper
set -e

echo "🚀 Running Comprehensive Search Validation Tests..."
echo "Activating runner environment..."

# Activate the runner environment
source /app/v.sh runner

echo "Virtual environment: $VIRTUAL_ENV"
echo "Working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"

# Run the comprehensive search validation test
cd /app
python test_comprehensive_search_validation.py