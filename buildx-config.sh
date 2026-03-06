#!/bin/bash

# Setup Docker buildx for multi-arch builds
# This script configures buildx builder for cross-platform image builds

set -e

echo "=== Docker BuildX Configuration ==="

# Check if buildx is available
if ! docker buildx version &>/dev/null; then
    echo "Error: Docker buildx is not available"
    exit 1
fi

# Create a new builder instance for multi-arch builds
echo "Creating buildx builder instance..."
docker buildx create --use --name llmmll-builder || true

# Enable buildx
docker buildx use llmmll-builder

# Inspect builder
echo "Builder info:"
docker buildx inspect --verbose

# Build multi-arch images
echo ""
echo "Usage examples:"
echo "  # Build for AMD64 only (server, composer)"
echo "  docker buildx build --platform linux/amd64 -t registry:image --push -f k8s/Dockerfile ."
echo ""
echo "  # Build for multiple architectures"
echo "  docker buildx build --platform linux/amd64,linux/arm64 -t registry:image --push -f k8s/Dockerfile ."
echo ""
echo "  # Build locally without pushing (for testing)"
echo "  docker buildx build --platform linux/amd64 -t local/image --load -f k8s/Dockerfile ."