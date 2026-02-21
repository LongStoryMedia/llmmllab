#!/bin/bash
# This script provides an easy way to enter your llmmll pod
# with the run_with_env.sh file automatically sourced

# Get the first pod name in the llmmll namespace
POD_NAME=$(kubectl get pods -n llmmll -o jsonpath='{.items[0].metadata.name}')


echo "Connecting to pod: $POD_NAME"

# Execute the kubectl command using the custom entrypoint
kubectl exec -it -n llmmll $POD_NAME -- /app/k8s/custom-entrypoint.sh

# Note: If you're using a different mount path for the scripts in the pod,
# you'll need to adjust the path to custom-entrypoint.sh accordingly
