#!/bin/bash
# Wrapper script for embedding smoke test that ensures proper environment setup

echo "🚀 Running Embedding Smoke Test with proper environment..."

# Use the runner environment
exec /app/v.sh runner python /app/embedding_smoke_test_core.py "$@"