#!/bin/bash

# Health check script for the gRPC-based inference API
# This script checks if both the Python gRPC server and Go maistro service are running correctly.

set -e # Exit immediately if a command exits with a non-zero status

# Default values
GRPC_HOST="localhost"
GRPC_PORT=50051
MAISTRO_HOST="localhost"
MAISTRO_PORT=8080
TIMEOUT=5

# Function to print usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --grpc-host HOST     gRPC server host (default: localhost)"
    echo "  --grpc-port PORT     gRPC server port (default: 50051)"
    echo "  --maistro-host HOST  Maistro service host (default: localhost)"
    echo "  --maistro-port PORT  Maistro service port (default: 8080)"
    echo "  --timeout SECONDS    Timeout in seconds (default: 5)"
    echo "  --help               Display this help message"
    exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
    --grpc-host)
        GRPC_HOST="$2"
        shift 2
        ;;
    --grpc-port)
        GRPC_PORT="$2"
        shift 2
        ;;
    --maistro-host)
        MAISTRO_HOST="$2"
        shift 2
        ;;
    --maistro-port)
        MAISTRO_PORT="$2"
        shift 2
        ;;
    --timeout)
        TIMEOUT="$2"
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

echo "=== Health Check for gRPC-based Inference API ==="
echo ""

# Check if the gRPC server is running
echo "Checking gRPC server at ${GRPC_HOST}:${GRPC_PORT}..."
if timeout ${TIMEOUT} bash -c "</dev/tcp/${GRPC_HOST}/${GRPC_PORT}" &>/dev/null; then
    echo "✅ gRPC server is running."
else
    echo "❌ gRPC server is not running or not reachable."
    exit 1
fi

# Check if the maistro service is running
echo "Checking maistro service at ${MAISTRO_HOST}:${MAISTRO_PORT}..."
if curl -s -o /dev/null -w "%{http_code}" http://${MAISTRO_HOST}:${MAISTRO_PORT}/health 2>/dev/null | grep -q "200"; then
    echo "✅ Maistro service is running."
else
    echo "❌ Maistro service is not running or not reachable."
    exit 1
fi

echo ""
echo "=== All services are healthy! ==="
