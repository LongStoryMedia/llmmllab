#!/bin/bash

set -e

# ---------------------------------------------
# Enhanced sync script
# Features added:
# 1. Propagate deletions from local debug/out to remote (without wiping new remote files)
# 2. Optional remote directory prune (--prune) to remove directories deleted locally even if they contained excluded files
# 3. Support multiple flags simultaneously (previous version only inspected $1)
# 4. Implement pull-only mode (-p / --pull-output)
# ---------------------------------------------

# Show help if requested
SHOW_HELP=0
WATCH_MODE=0
RESTART=0
PULL_ONLY=0
PRUNE_DIRS=0

for arg in "$@"; do
    case "$arg" in
        -w|--watch) WATCH_MODE=1 ;;
        -r|--restart) RESTART=1 ;;
        -p|--pull-output) PULL_ONLY=1 ;;
        -P|--prune) PRUNE_DIRS=1 ;;
        -h|--help) SHOW_HELP=1 ;;
        *) echo "Unknown option: $arg"; SHOW_HELP=1 ;;
    esac
done

if [ "$SHOW_HELP" = "1" ]; then
    echo "LLM ML Lab Code Sync Script"
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  (no args)          Full sync: pull benchmark data + debug output, propagate deletions, then push code"
    echo "  -p, --pull-output  Pull only benchmark + debug output (no code push)"
    echo "  -w, --watch        Watch for local changes and sync continuously"
    echo "  -r, --restart      Restart the ollama deployment after sync"
    echo "  -P, --prune        Prune remote directories deleted locally (force delete, even if non-empty)"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Deletion Propagation (debug/out):" 
    echo "  Local deletions in debug/out are detected via a manifest and removed remotely BEFORE pulling fresh output."
    echo "  New remote files are preserved (they are pulled down after deletions)."
    echo ""
    echo "Environment variables:" 
    echo "  REMOTE_NODE_HOST   Override default node host (default: lsnode-3)"
    echo "  REMOTE_NODE_USER   Override default node user (default: root)"
    echo ""
    echo "Advanced notes:" 
    echo "  --prune will compare directory trees and force remove remote directories not present locally (excluding safe list)."
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

DEBUG_OUT_LOCAL="${SCRIPT_DIR}/debug/out"
DEBUG_OUT_REMOTE="${NODE_CODE_PATH}/debug/out"
DEBUG_MANIFEST="${DEBUG_OUT_LOCAL}/.manifest"
# Persist backup outside the debug/out tree so a local wipe keeps history
SYNC_STATE_DIR="${SCRIPT_DIR}/.sync_state"
mkdir -p "${SYNC_STATE_DIR}" 2>/dev/null || true
DEBUG_MANIFEST_BACKUP="${SYNC_STATE_DIR}/debug_out.manifest.last"

# Function: propagate deletions from local debug/out to remote BEFORE pulling remote changes
propagate_debug_out_deletions() {
    local prev_manifest_path=""
    if [ -f "${DEBUG_MANIFEST}" ]; then
        prev_manifest_path="${DEBUG_MANIFEST}"
    elif [ -f "${DEBUG_MANIFEST_BACKUP}" ]; then
        prev_manifest_path="${DEBUG_MANIFEST_BACKUP}"
    fi
    [ -n "${prev_manifest_path}" ] || return 0

    # Build current list (empty if directory missing)
    local tmp_current
    tmp_current=$(mktemp)
    if [ -d "${DEBUG_OUT_LOCAL}" ]; then
        (cd "${DEBUG_OUT_LOCAL}" && find . -mindepth 1 ! -name '.manifest' ! -name 'debug_out.manifest.last' ! -name '.manifest.last' -print | sort > "${tmp_current}")
    else
        : > "${tmp_current}"
    fi

    # Normalize previous manifest (strip any legacy manifest/self entries)
    local tmp_prev
    tmp_prev=$(mktemp)
    grep -vE '^\./?\.manifest(\.last)?$' "${prev_manifest_path}" | grep -vE 'debug_out.manifest.last$' | sort > "${tmp_prev}" || true

    # Determine deletions = prev - current
    local deleted_list
    deleted_list=$(comm -23 "${tmp_prev}" "${tmp_current}")

    # If current is empty and previous had entries -> full wipe requested
    if [ ! -s "${tmp_current}" ] && [ -s "${tmp_prev}" ]; then
        deleted_list=$(cat "${tmp_prev}")
    fi

    if [ -n "${deleted_list}" ]; then
        echo "🗑  Propagating deletions to remote debug/out:" 
        while IFS= read -r rel; do
            [ -z "$rel" ] && continue
            rel_clean="${rel#./}"
            echo "   - deleting ${rel_clean}"
            ssh -o BatchMode=yes "${NODE_USER}@${NODE_HOST}" "rm -rf '${DEBUG_OUT_REMOTE}/${rel_clean}'" || echo "     (warn) failed to delete ${rel_clean}"
        done <<< "${deleted_list}"
    fi

    # Clean up
    rm -f "${tmp_current}" "${tmp_prev}"
}

# Function: update manifest after pulling remote debug/out
update_debug_manifest() {
    [ -d "${DEBUG_OUT_LOCAL}" ] || { rm -f "${DEBUG_MANIFEST}"; return 0; }
    local tmp_manifest
    tmp_manifest=$(mktemp)
    (cd "${DEBUG_OUT_LOCAL}" && find . -mindepth 1 ! -name '.manifest' ! -name '.manifest.last' ! -name 'debug_out.manifest.last' -print | sort > "${tmp_manifest}")
    mv "${tmp_manifest}" "${DEBUG_MANIFEST}"
    cp -f "${DEBUG_MANIFEST}" "${DEBUG_MANIFEST_BACKUP}" 2>/dev/null || true
}

# Function: prune remote directories removed locally (force delete non-empty)
prune_remote_directories() {
    echo "🌲 Pruning remote directories deleted locally..."
    local local_dirs_file remote_dirs_file prune_list
    local_dirs_file=$(mktemp)
    remote_dirs_file=$(mktemp)
    # Safe excludes (patterns relative to project root)
    local safe_prune_excludes=( './.git' './benchmark_data' './llama.cpp' )
    # Build local directory list
    ( cd "${SCRIPT_DIR}" && find . -type d | sort > "${local_dirs_file}" )
    # Build remote directory list
    ssh -o BatchMode=yes "${NODE_USER}@${NODE_HOST}" "cd '${NODE_CODE_PATH}' && find . -type d | sort" > "${remote_dirs_file}" || { echo "   (warn) could not list remote dirs"; return; }
    # Build prune list = remote minus local, excluding safe list
    prune_list=$(comm -23 "${remote_dirs_file}" "${local_dirs_file}")
    if [ -z "${prune_list}" ]; then
        echo "   No directories to prune"
    else
        while IFS= read -r dir; do
            [ -z "$dir" ] && continue
            skip=0
            for ex in "${safe_prune_excludes[@]}"; do
                if [ "$dir" = "$ex" ]; then
                    skip=1; break
                fi
            done
            [ $skip -eq 1 ] && continue
            echo "   - removing remote directory ${dir}"
            ssh -o BatchMode=yes "${NODE_USER}@${NODE_HOST}" "rm -rf '${NODE_CODE_PATH}/${dir#./}'" || echo "     (warn) failed to remove ${dir}"
        done <<< "${prune_list}"
    fi
    rm -f "${local_dirs_file}" "${remote_dirs_file}"
}

# 1. Propagate any deletions for debug/out BEFORE pulling (only on full sync path)
if [ "$PULL_ONLY" = "0" ]; then
    propagate_debug_out_deletions
fi

# Pull benchmark data from server (always, unless directory missing remotely)
echo "📊 Pulling benchmark data from server..."
rsync -avzru \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='llama.cpp/' \
    "${NODE_USER}@${NODE_HOST}:${NODE_CODE_PATH}/benchmark_data/" "${SCRIPT_DIR}/benchmark_data/" 2>/dev/null || echo "   (No remote benchmark_data yet)"

# Pull debug output files from server (after deletion propagation)
echo "🔍 Pulling debug output files from server..."
rsync -avzru \
    --include='*.json' \
    --include='*.txt' \
    --include='*.log' \
    --include='*/' \
    --exclude='*' \
    "${NODE_USER}@${NODE_HOST}:${NODE_CODE_PATH}/debug/out/" "${SCRIPT_DIR}/debug/out/" 2>/dev/null || echo "   (No debug output files found on server yet)"

# Update manifest (captures new remote state)
update_debug_manifest

if [ "$PULL_ONLY" = "0" ]; then
    echo "📤 Pushing code changes to server..."
    if [ "$PRUNE_DIRS" = "1" ]; then
        # Without --delete; prune handles removals
        rsync -avzru \
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
    else
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
    fi
    echo "✅ Code synced successfully"
    if [ "$PRUNE_DIRS" = "1" ]; then
        prune_remote_directories
    fi
else
    echo "ℹ️  Pull-only mode: skipping code push"
fi

# Check if we should watch for changes and continuously sync (full sync only)
if [ "$WATCH_MODE" = "1" ] && [ "$PULL_ONLY" = "0" ]; then
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
        echo "✅ Code synced at $(date)"
    done
fi

# Optionally restart the deployment
if [ "$RESTART" = "1" ]; then
    echo "Restarting ollama deployment..."
    kubectl rollout restart deployment ollama -n ollama
    echo "Deployment restarted. It may take a moment to become available."
fi
