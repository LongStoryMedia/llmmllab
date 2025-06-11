#!/bin/bash

source "$(dirname "$0")/../helpers.sh"

set -e

# Use registry.local instead of NODE_IP:PORT
REGISTRY_HOME="$(dirname "$0")/../registry"
REGISTRY_URL="192.168.0.71:31500"
VERSION="$(date +%Y.%m.%d)"

# Registry credentials
USER_SECRET_FILE="${REGISTRY_HOME}/.secrets/registryuser"
PW_SECRET_FILE="${REGISTRY_HOME}/.secrets/registrypw"

if [[ -f "$USER_SECRET_FILE" && -f "$PW_SECRET_FILE" ]]; then
    REGISTRY_USER=$(cat "$USER_SECRET_FILE")
    REGISTRY_PW=$(cat "$PW_SECRET_FILE")
else
    echo "Registry credentials not found. Please run registry-mgmt.sh install first."
    exit 1
fi

# Create namespace
kubectl create namespace proxyllama || true

# Create secrets for DB access
kubectl create secret generic secrets \
    -n proxyllama \
    --from-file=psqlpw="${1}/../psql/.secrets/psqlpw" \
    --dry-run=client -o yaml | kubectl apply -f - --wait=true

# Create registry credentials secret
kubectl create secret docker-registry registry-credentials \
    -n proxyllama \
    --docker-server="${REGISTRY_URL}" \
    --docker-username="${REGISTRY_USER}" \
    --docker-password="${REGISTRY_PW}" \
    --dry-run=client -o yaml | kubectl apply -f - --wait=true

echo "Building and pushing image to private registry..."
bash "$(dirname "$0")/build-push.sh" "${VERSION}"

# Update the deployment with the correct registry URL
sed -e "s/\${REGISTRY_URL}/${REGISTRY_URL}/g" -e "s/\${VERSION}/\"${VERSION}\"/g" "$(dirname "$0")/k8s/deployment.yaml" >"$(dirname "$0")/k8s/deployment.${VERSION}.yaml"

echo "Applying Kubernetes resources..."
kubectl apply -n proxyllama -f "$(dirname "$0")/k8s/deployment.${VERSION}.yaml" --wait=true
kubectl apply -n proxyllama -f "$(dirname "$0")/k8s/service.yaml" --wait=true
kubectl apply -n proxyllama -f "$(dirname "$0")/k8s/referencegrant.yaml" --wait=true

echo "✅ ProxyLlama deployment complete using private registry at ${REGISTRY_URL}"
