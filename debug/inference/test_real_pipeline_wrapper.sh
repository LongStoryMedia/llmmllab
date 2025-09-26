#!/bin/bash

# Real End-to-End Pipeline Test Wrapper
echo "🚀 Running Real End-to-End Pipeline Test..."

# Activate the runner environment and run the test
exec /app/v.sh runner python /app/test_real_end_to_end_pipeline.py