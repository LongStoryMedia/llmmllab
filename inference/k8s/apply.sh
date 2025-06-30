#!/bin/bash

set -e

# Create the namespace if it doesn't exist
kubectl create namespace ollama || true

# get rabbitmq pw secret
RABBITMQ_PASSWORD=$(kubectl get secret secrets -n rabbitmq -o jsonpath='{.data.rabbitmqpw}' | base64 --decode)

# Create secrets for RabbitMQ access
kubectl create secret generic rabbitmq \
    -n ollama \
    --from-literal=password="$RABBITMQ_PASSWORD" \
    --dry-run=client -o yaml | kubectl apply -f - --wait=true

# Create secrets for DB access
kubectl create secret generic hf-token \
    -n ollama \
    --from-file=token="$(dirname "$0")/.secrets/hf-token" \
    --dry-run=client -o yaml | kubectl apply -f - --wait=true

echo "Applying PVC..."
kubectl apply -f "$(dirname "$0")/pvc.yaml" -n ollama --wait=true

echo "Applying init script ConfigMap..."
kubectl apply -f "$(dirname "$0")/init-script.yaml" -n ollama --wait=true

echo "Applying Ollama deployment..."
kubectl apply -f "$(dirname "$0")/deployment.yaml" -n ollama --wait=true

echo "Applying Ollama service..."
kubectl apply -f "$(dirname "$0")/service.yaml" -n ollama --wait=true

kubectl apply -f "$(dirname "$0")/referencegrant.yaml"

echo "Ollama deployment complete! Service is available at ollama.ollama.svc.cluster.local:11434"
echo "Wait a few minutes for the models to be loaded and configured."
