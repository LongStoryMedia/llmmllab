#!/bin/bash
set -e

# Migration script: ollama namespace → llmmll namespace
# Preserves all PVC data by renaming hostPath directories on the node
#
# Prerequisites:
#   - kubectl access to the cluster
#   - SSH access to lsnode-3 (the node hosting the volumes)
#
# What this does:
#   1. Scales down the deployment in the old namespace
#   2. Moves hostPath directories on the node (data preserved)
#   3. Removes old PVs/PVCs (finalizers cleared to avoid stuck resources)
#   4. Deletes the old namespace
#   5. Runs apply.sh to create the new namespace and all resources
#
# The /root/.ollama mountPath inside the container is intentionally preserved
# for future ollama binary usage.

NODE_HOST="lsnode-3.local"
OLD_NS="ollama"
NEW_NS="llmmll"

echo "=== Migrating namespace: $OLD_NS → $NEW_NS ==="
echo ""

# Step 1: Scale down the deployment
echo "Step 1: Scaling down deployment in $OLD_NS namespace..."
kubectl scale deployment ollama -n "$OLD_NS" --replicas=0 2>/dev/null || echo "  (deployment not found or already scaled down)"
echo "  Waiting for pod termination..."
kubectl wait --for=delete pod -l app=ollama -n "$OLD_NS" --timeout=120s 2>/dev/null || true
echo "  ✅ Deployment scaled down"
echo ""

# Step 2: Rename hostPath directories on the node to preserve data
echo "Step 2: Renaming hostPath directories on $NODE_HOST..."
ssh root@"$NODE_HOST" bash -s <<'REMOTE_COMMANDS'
set -e

# Only the ollama-models directory needs renaming
# Other PVCs (generated-images, code-base, models) keep their paths
if [ -d /data/ollama-models ]; then
    echo "  Moving /data/ollama-models → /data/llmmll-models"
    mv /data/ollama-models /data/llmmll-models
    echo "  ✅ Directory renamed"
elif [ -d /data/llmmll-models ]; then
    echo "  /data/llmmll-models already exists, skipping"
else
    echo "  /data/ollama-models not found, creating /data/llmmll-models"
    mkdir -p /data/llmmll-models
fi
REMOTE_COMMANDS
echo ""

# Step 3: Remove old PVs and PVCs (clear finalizers to avoid stuck state)
echo "Step 3: Removing old PVs and PVCs..."
for RESOURCE in ollama-models generated-images code-base models; do
    echo "  Cleaning up PVC/$RESOURCE in $OLD_NS..."
    kubectl patch pvc "$RESOURCE" -n "$OLD_NS" -p '{"metadata":{"finalizers":null}}' 2>/dev/null || true
    kubectl delete pvc "$RESOURCE" -n "$OLD_NS" --grace-period=0 --force 2>/dev/null || true

    echo "  Cleaning up PV/$RESOURCE..."
    kubectl patch pv "$RESOURCE" -p '{"metadata":{"finalizers":null}}' 2>/dev/null || true
    kubectl delete pv "$RESOURCE" --grace-period=0 --force 2>/dev/null || true
done
echo "  ✅ Old PVs/PVCs removed"
echo ""

# Step 4: Remove old secrets, configmaps, services, deployments
echo "Step 4: Cleaning up remaining resources in $OLD_NS..."
kubectl delete deployment ollama -n "$OLD_NS" 2>/dev/null || true
kubectl delete service ollama -n "$OLD_NS" 2>/dev/null || true
kubectl delete configmap ollama-init-script -n "$OLD_NS" 2>/dev/null || true
kubectl delete referencegrant access-to-ollama -n "$OLD_NS" 2>/dev/null || true
for SECRET in rabbitmq auth-client db-credentials internal-api-key hf-token; do
    kubectl delete secret "$SECRET" -n "$OLD_NS" 2>/dev/null || true
done
echo "  ✅ Old resources removed"
echo ""

# Step 5: Delete old namespace
echo "Step 5: Deleting old namespace $OLD_NS..."
kubectl delete namespace "$OLD_NS" --timeout=60s 2>/dev/null || true
echo "  ✅ Old namespace deleted"
echo ""

# Step 6: Deploy to new namespace using apply.sh
echo "Step 6: Deploying to new namespace $NEW_NS..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"$SCRIPT_DIR/apply.sh"
echo ""

echo "=== Migration complete! ==="
echo ""
echo "New service endpoint: llmmll.llmmll.svc.cluster.local:11434"
echo ""
echo "Verify with:"
echo "  kubectl get pods -n $NEW_NS"
echo "  kubectl get pvc -n $NEW_NS"
echo "  kubectl logs -f -n $NEW_NS -l app=$NEW_NS"
