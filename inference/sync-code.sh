#!/bin/bash

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Node details - update these with your specific node information
NODE_USER="root"
NODE_HOST="lsnode-3"
NODE_CODE_PATH="/data/code-base"

# Check if NODE_HOST environment variable is set, otherwise use default
if [ -n "${REMOTE_NODE_HOST}" ]; then
    NODE_HOST="${REMOTE_NODE_HOST}"
fi

# Check if NODE_USER environment variable is set, otherwise use default
if [ -n "${REMOTE_NODE_USER}" ]; then
    NODE_USER="${REMOTE_NODE_USER}"
fi

echo "Syncing code to ${NODE_USER}@${NODE_HOST}:${NODE_CODE_PATH}..."

# Create the target directory if it doesn't exist
ssh ${NODE_USER}@${NODE_HOST} "mkdir -p ${NODE_CODE_PATH}"

# Use rsync to sync the local code to the remote node
rsync -avz --delete \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='llama.cpp/' \
    "${SCRIPT_DIR}/" "${NODE_USER}@${NODE_HOST}:${NODE_CODE_PATH}/"

echo "✅ Code synced successfully"

# Now make sure the directory structure is correct - app.py needs to be directly in the mounted directory
echo "Ensuring correct file structure on remote node..."
ssh ${NODE_USER}@${NODE_HOST} "cd ${NODE_CODE_PATH} && ls -la && echo 'Verifying app.py exists:' && if [ -f app.py ]; then echo 'app.py found at root level'; else echo 'ERROR: app.py not found at root level'; fi"

# Check if we should watch for changes and continuously sync
if [ "$1" = "--watch" ] || [ "$1" = "-w" ]; then
    echo "Watching for changes and syncing continuously. Press Ctrl+C to stop."

    # Check if fswatch is installed
    if ! command -v fswatch &>/dev/null; then
        echo "fswatch not found. Please install it with 'brew install fswatch' to use watch mode."
        exit 1
    fi

    fswatch -o "${SCRIPT_DIR}" | while read f; do
        echo "Change detected, syncing..."
        rsync -avzr "${NODE_USER}@${NODE_HOST}:${NODE_CODE_PATH}/config" "${SCRIPT_DIR}/s-config"

        rsync -avz --delete \
            --exclude='.git/' \
            --exclude='.venv/' \
            --exclude='venv/' \
            --exclude='__pycache__/' \
            --exclude='*.pyc' \
            "${SCRIPT_DIR}/" "${NODE_USER}@${NODE_HOST}:${NODE_CODE_PATH}/"
        echo "✅ Code synced at $(date)"
    done
fi

# Optionally restart the deployment
if [ "$1" = "--restart" ] || [ "$1" = "-r" ]; then
    echo "Restarting ollama deployment..."
    kubectl rollout restart deployment ollama -n ollama
    echo "Deployment restarted. It may take a moment to become available."
fi
