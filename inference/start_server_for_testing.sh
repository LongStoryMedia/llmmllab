#!/bin/bash
# Script to start the FastAPI server for testing API versioning

echo "Starting FastAPI server for testing API versioning..."
echo "Press Ctrl+C to stop the server"

# Determine the Python executable to use
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found"
    exit 1
fi

# Start the server using uvicorn
$PYTHON_CMD -m uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
