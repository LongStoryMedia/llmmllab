#!/bin/bash

# Delete the existing PVC and PV for code-base
echo "Deleting existing code-base PVC and PV..."
kubectl delete pvc code-base -n ollama --wait=false
kubectl delete pv code-base --wait=false

# Wait for resources to be deleted
echo "Waiting for resources to be deleted..."
kubectl wait --for=delete pvc/code-base -n ollama --timeout=60s || true
kubectl wait --for=delete pv/code-base --timeout=60s || true

# Verify that resources are deleted
if kubectl get pv code-base 2>/dev/null; then
    echo "Warning: PV code-base still exists. You may need to force delete it."
    echo "Run: kubectl patch pv code-base -p '{\"metadata\":{\"finalizers\":null}}'"
    echo "Then: kubectl delete pv code-base --grace-period=0 --force"
else
    echo "✅ PV code-base deleted successfully"
fi

if kubectl get pvc code-base -n ollama 2>/dev/null; then
    echo "Warning: PVC code-base still exists. You may need to force delete it."
    echo "Run: kubectl patch pvc code-base -n ollama -p '{\"metadata\":{\"finalizers\":null}}'"
    echo "Then: kubectl delete pvc code-base -n ollama --grace-period=0 --force"
else
    echo "✅ PVC code-base deleted successfully"
fi

echo "Now you can reapply the PVC configuration with: kubectl apply -f pvc.yaml"
