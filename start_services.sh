#!/bin/bash

# Script to start both the Python gRPC server and Go maistro service
# This script is useful for development and testing

set -e # Exit immediately if a command exits with a non-zero status

# Default values
GRPC_PORT=50051
MAISTRO_PORT=8080

# Function to print usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --grpc-port PORT     gRPC server port (default: 50051)"
    echo "  --maistro-port PORT  Maistro service port (default: 8080)"
    echo "  --help               Display this help message"
    exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
    --grpc-port)
        GRPC_PORT="$2"
        shift 2
        ;;
    --maistro-port)
        MAISTRO_PORT="$2"
        shift 2
        ;;
    --help)
        usage
        ;;
    *)
        echo "Unknown option: $1"
        usage
        ;;
    esac
done

# Function to handle CTRL+C
cleanup() {
    echo "Stopping services..."
    kill $GRPC_PID 2>/dev/null || true
    kill $MAISTRO_PID 2>/dev/null || true
    exit 0
}

# Register the cleanup function for SIGINT (CTRL+C)
trap cleanup SIGINT

echo "=== Starting Services ==="
echo ""

# Generate proto code if needed
echo "Generating proto code..."
./generate_proto.sh

# Start the gRPC server
echo "Starting gRPC server on port ${GRPC_PORT}..."
python inference/grpc_server/server.py --port ${GRPC_PORT} &
GRPC_PID=$!
echo "gRPC server started with PID ${GRPC_PID}"

# Wait a moment to ensure the gRPC server is ready
sleep 2

# Set environment variables for maistro
export INFERENCE_USE_GRPC=true
export INFERENCE_GRPC_ADDRESS=localhost:${GRPC_PORT}
export INFERENCE_USE_SECURE_CONNECTION=false

# Start the maistro service
echo "Starting maistro service on port ${MAISTRO_PORT}..."
cd maistro
go run main.go --port ${MAISTRO_PORT} &
MAISTRO_PID=$!
cd ..
echo "Maistro service started with PID ${MAISTRO_PID}"

echo ""
echo "=== Services Started ==="
echo "gRPC server: http://localhost:${GRPC_PORT}"
echo "Maistro service: http://localhost:${MAISTRO_PORT}"
echo ""
echo "Press CTRL+C to stop both services"

# Wait for both processes to finish (or until CTRL+C)
wait
