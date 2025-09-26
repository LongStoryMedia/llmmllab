#!/bin/bash

set -e

# Show help if requested
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "LLM ML Lab Code Sync Script"
    echo "Usage: $0 [option]"
    echo ""
    echo "Options:"
    echo "  (no args)        Full sync: pull benchmark data + debug output, then push code"
    echo "  -w, --watch      Watch for local changes and sync continuously" 
    echo "  -r, --restart    Restart the ollama deployment after sync"
    echo "  -p, --pull-output Pull only output files (benchmark data + debug output)"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  REMOTE_NODE_HOST Override default node host (default: lsnode-3)"
    echo "  REMOTE_NODE_USER Override default node user (default: root)"
    exit 0
fi

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

# Pull benchmark data from server
echo "📊 Pulling benchmark data from server..."
rsync -avzru \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='llama.cpp/' \
    "${NODE_USER}@${NODE_HOST}:${NODE_CODE_PATH}/benchmark_data/" "${SCRIPT_DIR}/benchmark_data/"

# Pull debug output files from server
echo "🔍 Pulling debug output files from server..."
rsync -avzru \
    --include='*.json' \
    --include='*.txt' \
    --include='*.log' \
    --exclude='*' \
    "${NODE_USER}@${NODE_HOST}:${NODE_CODE_PATH}/debug/out/" "${SCRIPT_DIR}/debug/out/" 2>/dev/null || echo "   (No debug output files found on server yet)"

# Use rsync to sync the local code to the remote node
echo "📤 Pushing code changes to server..."
rsync -avzru --delete \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='llama.cpp/' \
    --exclude='benchmark_data/' \
    --exclude='debug/out/' \
    --exclude='.pytest_cache/' \
    --exclude='.DS_Store' \
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
        echo "Change detected in ${f}, syncing..."
        rsync -avruz --delete \
            --exclude='.git/' \
            --exclude='.venv/' \
            --exclude='venv/' \
            --exclude='__pycache__/' \
            --exclude='*.pyc' \
            --exclude='llama.cpp/' \
            --exclude='benchmark_data/' \
            --exclude='debug/out/' \
            --exclude='.pytest_cache/' \
            --exclude='.DS_Store' \
            "${SCRIPT_DIR}/" "${NODE_USER}@${NODE_HOST}:${NODE_CODE_PATH}/"
        
        # Also pull any new output files
        rsync -avzru \
            --include='*.json' \
            --include='*.txt' \
            --include='*.log' \
            --exclude='*' \
            "${NODE_USER}@${NODE_HOST}:${NODE_CODE_PATH}/debug/out/" "${SCRIPT_DIR}/debug/out/" 2>/dev/null
        
        echo "✅ Code synced at $(date)"
    done
fi

# Pull output files only (useful after running tests)
if [ "$1" = "--pull-output" ] || [ "$1" = "-p" ]; then
    echo "📋 Pulling only output files from server..."
    
    # Pull benchmark data
    echo "📊 Pulling benchmark data..."
    rsync -avzru \
        "${NODE_USER}@${NODE_HOST}:${NODE_CODE_PATH}/benchmark_data/" "${SCRIPT_DIR}/benchmark_data/"
    
    # Pull debug output files  
    echo "🔍 Pulling debug output files..."
    rsync -avzru \
        --include='*.json' \
        --include='*.txt' \
        --include='*.log' \
        --exclude='*' \
        "${NODE_USER}@${NODE_HOST}:${NODE_CODE_PATH}/debug/out/" "${SCRIPT_DIR}/debug/out/"
    
    echo "✅ Output files pulled successfully"
    exit 0
fi

# Optionally restart the deployment
if [ "$1" = "--restart" ] || [ "$1" = "-r" ]; then
    echo "Restarting ollama deployment..."
    kubectl rollout restart deployment ollama -n ollama
    echo "Deployment restarted. It may take a moment to become available."
fi
