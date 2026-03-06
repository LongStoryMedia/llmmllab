#!/bin/bash

set -e

# Usage: ./build-image.sh <service> <platform>
# <service>: runner, server, composer
# <platform>: multi-arch (default), or lsnode-3 (for GPU-specific builds)

REGISTRY=${REGISTRY:-192.168.0.71:31500}
TAG=${TAG:-latest}

SERVICE=$1
PLATFORM=${2:-multi-arch}

if [ -z "$SERVICE" ]; then
    echo "Usage: $0 <service> [platform]"
    echo "  service: runner, server, composer"
    echo "  platform: multi-arch (default), lsnode-3"
    exit 1
fi

if [ "$SERVICE" != "runner" ] && [ "$SERVICE" != "server" ] && [ "$SERVICE" != "composer" ]; then
    echo "Error: Service must be runner, server, or composer"
    exit 1
fi

if [ "$PLATFORM" != "multi-arch" ] && [ "$PLATFORM" != "lsnode-3" ]; then
    echo "Error: Platform must be multi-arch or lsnode-3"
    exit 1
fi

echo "Building $SERVICE image with platform: $PLATFORM"

if [ "$PLATFORM" = "lsnode-3" ]; then
    # GPU-specific build on lsnode-3
    echo "Building $SERVICE image on lsnode-3 (AMD64)..."
    ssh root@lsnode-3.local "
      TEMP_DIR=\$(mktemp -d)
      trap 'rm -rf \${TEMP_DIR}' EXIT
      echo \"Created temp directory: \${TEMP_DIR}\"

      echo \"Syncing code to temp directory...\"
      cp -r /data/code-base/* \${TEMP_DIR}/

      echo \"Building $SERVICE image...\"
      cd \${TEMP_DIR}/$SERVICE && docker build -t \${REGISTRY}/$SERVICE:\${TAG} -f k8s/Dockerfile . --push

      echo \"$SERVICE image built and pushed: \${REGISTRY}/$SERVICE:\${TAG}\"
    "
else
    # Multi-arch build
    echo "Building $SERVICE multi-arch image (linux/amd64, linux/arm64)..."
    cd $SERVICE/k8s
    docker buildx build \
        --platform linux/amd64,linux/arm64 \
        -t ${REGISTRY}/$SERVICE:${TAG} \
        --push \
        -f Dockerfile .
fi

echo "$SERVICE build complete: ${REGISTRY}/$SERVICE:${TAG}"