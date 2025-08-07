#!/bin/bash

set -e

# Set default tag if not provided
DOCKER_TAG=${DOCKER_TAG:-latest}
# Replace slashes with dots in the tag name for Docker compatibility
DOCKER_TAG=$(echo "$DOCKER_TAG" | tr '/' '.')
echo "Deploying inference with tag: $DOCKER_TAG"

# Create the namespace if it doesn't exist
kubectl create namespace ollama || true

# get rabbitmq pw secret
RABBITMQ_PASSWORD=$(kubectl get secret secrets -n rabbitmq -o jsonpath='{.data.rabbitmqpw}' | base64 --decode)

# Create secrets for RabbitMQ access
kubectl create secret generic rabbitmq \
-n ollama \
--from-literal=password="$RABBITMQ_PASSWORD" \
--dry-run=client -o yaml | kubectl apply -f - --wait=true

# Create secrets for auth client
kubectl create secret generic auth-client \
-n ollama \
--from-file=client_secret="$(dirname "$0")/.secrets/auth_client_secret" \
--dry-run=client -o yaml | kubectl apply -f - --wait=true

# Create secrets for DB access
kubectl create secret generic db-credentials \
-n ollama \
--from-file=password="$(dirname "$0")/.secrets/db_password" \
--dry-run=client -o yaml | kubectl apply -f - --wait=true

# Create secrets for HF token
kubectl create secret generic hf-token \
-n ollama \
--from-file=token="$(dirname "$0")/.secrets/hf-token" \
--dry-run=client -o yaml | kubectl apply -f - --wait=true

# Create secrets for Google search API key
kubectl create secret generic google-search-api-key \
-n ollama \
--from-file=key="$(dirname "$0")/.secrets/google-search-api-key" \
--dry-run=client -o yaml | kubectl apply -f - --wait=true

# Create secrets for google search CX
kubectl create secret generic google-search-cx \
-n ollama \
--from-file=key="$(dirname "$0")/.secrets/google-search-cx" \
--dry-run=client -o yaml | kubectl apply -f - --wait=true

if [ ! -d "$(dirname "$0")/.secrets" ]; then
    mkdir -p "$(dirname "$0")/.secrets"
fi

if [ ! -f "$(dirname "$0")/.secrets/internal-api-key" ]; then
    openssl rand -hex 16 > "$(dirname "$0")/.secrets/internal-api-key"
fi

# Create secrets for internal API access
kubectl create secret generic internal-api-key \
-n ollama \
--from-file=api_key="$(dirname "$0")/.secrets/internal-api-key" \
--dry-run=client -o yaml | kubectl apply -f - --wait=true

echo "Applying PVC..."
kubectl apply -f "$(dirname "$0")/pvc.yaml" -n ollama --wait=true

echo "Applying init script ConfigMap..."
kubectl apply -f "$(dirname "$0")/init-script.yaml" -n ollama --wait=true

echo "Updating deployment image to use tag: $DOCKER_TAG"
# Create a temporary file with the updated image tag
sed "s|image: 192.168.0.71:31500/inference:latest|image: 192.168.0.71:31500/inference:$DOCKER_TAG|g" "$(dirname "$0")/deployment.yaml" > "$(dirname "$0")/deployment.yaml.tmp"
mv "$(dirname "$0")/deployment.yaml.tmp" "$(dirname "$0")/deployment.yaml"

echo "Applying Ollama deployment..."
kubectl apply -f "$(dirname "$0")/deployment.yaml" -n ollama --wait=true

echo "Applying Ollama service..."
kubectl apply -f "$(dirname "$0")/service.yaml" -n ollama --wait=true

kubectl apply -f "$(dirname "$0")/referencegrant.yaml"

echo "Ollama deployment complete! Service is available at ollama.ollama.svc.cluster.local:11434"
echo "Wait a few minutes for the models to be loaded and configured."
