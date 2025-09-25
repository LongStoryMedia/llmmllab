#!/bin/bash
# Kubernetes Embedding Test Runner
# Run this script inside the Kubernetes pod to test the embedding pipeline

set -e

echo "🚀 Starting Embedding Pipeline Tests on Kubernetes Pod"
echo "=================================================="

# Get pod and environment info
echo "📋 Environment Information:"
echo "  Pod Name: ${HOSTNAME:-Unknown}"
echo "  Working Directory: $(pwd)"
echo "  Python Version: $(python --version 2>&1 || echo 'Not available')"
echo "  Available GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'No GPU detected')"
echo ""

# Check if we're in the right environment
if [[ ! -f "/app/v.sh" ]]; then
    echo "❌ Error: This script should be run inside the Kubernetes pod"
    echo "   Expected /app/v.sh to exist"
    exit 1
fi

# Set up the environment using the pod's virtual environment system
echo "🔧 Setting up environment..."

# Use the runner environment for embedding tests
export ENVIRONMENT="runner"

# Run the embedding tests using the pod's environment setup
echo "🧪 Running embedding pipeline tests..."
/app/v.sh runner python /app/test_embedding_pipeline.py

# Capture exit code
TEST_EXIT_CODE=$?

echo ""
echo "=================================================="
echo "🏁 Embedding Pipeline Tests Completed"

if [[ $TEST_EXIT_CODE -eq 0 ]]; then
    echo "✅ Status: ALL TESTS PASSED"
elif [[ $TEST_EXIT_CODE -eq 1 ]]; then
    echo "❌ Status: CRITICAL ISSUES DETECTED"
elif [[ $TEST_EXIT_CODE -eq 2 ]]; then
    echo "⚠️  Status: DEGRADED PERFORMANCE"
else
    echo "💥 Status: TEST EXECUTION FAILED"
fi

echo "Exit Code: $TEST_EXIT_CODE"
echo ""

# Additional diagnostics if tests failed
if [[ $TEST_EXIT_CODE -ne 0 ]]; then
    echo "🔍 Additional Diagnostics:"
    
    # Check for common issues
    echo "  Model files in /app/models:"
    ls -la /app/models/ 2>/dev/null | head -10 || echo "    No /app/models directory found"
    
    echo "  Memory usage:"
    free -h 2>/dev/null || echo "    Memory info not available"
    
    echo "  GPU status:"
    nvidia-smi 2>/dev/null | head -20 || echo "    No GPU information available"
    
    echo "  Recent logs (last 20 lines):"
    tail -20 embedding_test_results_*.json 2>/dev/null || echo "    No test results file found"
fi

exit $TEST_EXIT_CODE