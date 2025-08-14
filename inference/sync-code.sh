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

# rsync -avzru \
    # --exclude='.git/' \
    # --exclude='.venv/' \
    # --exclude='venv/' \
    # --exclude='__pycache__/' \
    # --exclude='*.pyc' \
    # --exclude='llama.cpp/' \
    # "${NODE_USER}@${NODE_HOST}:${NODE_CODE_PATH}/benchmark_data/" "${SCRIPT_DIR}/benchmark_data/"

# Use rsync to sync the local code to the remote node
rsync -avzru --delete \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='llama.cpp/' \
    --exclude='benchmark_data/' \
    --exclude='.pytest_cache/' \
    "${SCRIPT_DIR}/" "${NODE_USER}@${NODE_HOST}:${NODE_CODE_PATH}/"

echo "✅ Code synced successfully"

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
        rsync -avruz --delete \
            --exclude='.git/' \
            --exclude='.venv/' \
            --exclude='venv/' \
            --exclude='__pycache__/' \
            --exclude='*.pyc' \
            --exclude='llama.cpp/' \
            --exclude='benchmark_data/' \
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
