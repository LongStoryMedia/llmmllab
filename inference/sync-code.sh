#!/bin/bash
# Sync the inference/ tree to the remote k8s node.
#
# Flags:
#   -w, --watch    Watch for local changes and sync continuously.
#   -r, --restart  Restart the llmmll deployment after sync.

set -e

WATCH_MODE=0
RESTART=0
for arg in "$@"; do
    case "$arg" in
        -w|--watch) WATCH_MODE=1 ;;
        -r|--restart) RESTART=1 ;;
        -h|--help)
            cat <<EOF
Usage: $0 [options]
  -w, --watch    Watch for local changes and sync continuously
  -r, --restart  Restart the llmmll deployment after sync
Env:
  REMOTE_NODE_HOST, REMOTE_NODE_USER override node defaults.
EOF
            exit 0
        ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_USER="${REMOTE_NODE_USER:-root}"
NODE_HOST="${REMOTE_NODE_HOST:-lsnode-3.local}"
NODE_CODE_PATH="/data/code-base"

RSYNC_EXCLUDES=(
    --exclude='.git/'
    --exclude='.venv/'
    --exclude='venv/'
    --exclude='__pycache__/'
    --exclude='*.pyc'
    --exclude='llama.cpp/'
    --exclude='.pytest_cache/'
    --exclude='.DS_Store'
    --exclude='benchmark_data/'
    --exclude='evaluation/'
    --exclude='.env'
)

sync_once() {
    echo "📤 Syncing → ${NODE_USER}@${NODE_HOST}:${NODE_CODE_PATH}"
    rsync -avzru --delete "${RSYNC_EXCLUDES[@]}" \
    "${SCRIPT_DIR}/" "${NODE_USER}@${NODE_HOST}:${NODE_CODE_PATH}/"
    echo "✅ Synced"
    if [ "$RESTART" = "1" ]; then
        kubectl rollout restart deployment llmmll -n llmmll
    fi
}

if [ "$WATCH_MODE" = "1" ]; then
    if ! command -v fswatch >/dev/null 2>&1; then
        echo "fswatch not found — install with: brew install fswatch"
        exit 1
    fi
    sync_once
    echo "👀 Watching ${SCRIPT_DIR} for changes..."
    fswatch -o "${SCRIPT_DIR}" --exclude='\.git' --exclude='__pycache__' --exclude='\.pyc$' | \
    while read -r _; do
        echo "Change detected, syncing..."
        sync_once
    done
else
    sync_once
fi
