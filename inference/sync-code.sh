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
RESET_DEBUG_OUT=0

for arg in "$@"; do
    case "$arg" in
        -w|--watch) WATCH_MODE=1 ;;
        -r|--restart) RESTART=1 ;;
        -p|--pull-output) PULL_ONLY=1 ;;
        -P|--prune) PRUNE_DIRS=1 ;;
    -R|--reset-debug-out) RESET_DEBUG_OUT=1 ;;
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
    echo "  -R, --reset-debug-out  Force remote debug/out to be fully cleared before pull"
    echo "  -h, --help             Show this help message"
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

    # If explicit reset flag, nuke remote dir except .gitignore and return
    if [ "${RESET_DEBUG_OUT}" = "1" ]; then
        echo "🧨 Resetting remote debug/out (full wipe)"
        ssh -o BatchMode=yes "${NODE_USER}@${NODE_HOST}" "rm -rf '${DEBUG_OUT_REMOTE}' && mkdir -p '${DEBUG_OUT_REMOTE}'" || echo "   (warn) remote reset failed"
        return 0
    fi

    # Build deletion list by walking previous manifest and checking local existence
    local to_delete_file line rel_clean local_path had_prev=0 del_count=0 keep_count=0
    to_delete_file=$(mktemp)
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        # Skip manifest bookkeeping entries
        if [[ "$line" =~ \\.manifest ]]; then
            continue
        fi
        had_prev=1
        rel_clean="${line#./}"
        local_path="${DEBUG_OUT_LOCAL}/${rel_clean}"
        if [ ! -e "${local_path}" ]; then
            echo "${rel_clean}" >> "${to_delete_file}"
            del_count=$((del_count+1))
        else
            keep_count=$((keep_count+1))
        fi
    done < "${prev_manifest_path}"

    # Full wipe heuristic: local directory missing OR only .gitignore present and deletions less than 80% of previous -> treat as full wipe
    if [ "$had_prev" = "1" ]; then
        local total_prev=$((del_count+keep_count))
        if { [ ! -d "${DEBUG_OUT_LOCAL}" ] || [ "$keep_count" -le 1 ]; } && [ "$total_prev" -gt 0 ] && [ $del_count -lt $total_prev ]; then
            # Rebuild deletion list as everything
            cat /dev/null > "${to_delete_file}"
            while IFS= read -r line; do
                [ -z "$line" ] && continue
                if [[ "$line" =~ \\.manifest ]]; then
                    continue
                fi
                echo "${line#./}" >> "${to_delete_file}"
            done < "${prev_manifest_path}"
            del_count=$total_prev
            keep_count=0
        fi
    fi

    if [ "$del_count" -gt 0 ]; then
        echo "🗑  Propagating deletions to remote debug/out (${del_count} files)..."
        while IFS= read -r rel_clean; do
            [ -z "$rel_clean" ] && continue
            echo "   - deleting ${rel_clean}"
            ssh -o BatchMode=yes "${NODE_USER}@${NODE_HOST}" "rm -rf '${DEBUG_OUT_REMOTE}/${rel_clean}'" || echo "     (warn) failed to delete ${rel_clean}"
        done < "${to_delete_file}"
    fi
    [ -n "${SYNC_DEBUG}" ] && echo "   (debug) deletions=${del_count} kept=${keep_count}"
    rm -f "${to_delete_file}"
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

    # Safe excludes (exact or prefix). These should NEVER be pruned.
    local safe_prune_excludes=( './.git' './benchmark_data' './llama.cpp' './.venv' './venv' './env' './envs' )

    # Helper: return 0 (true) if path should be skipped (env / internal / python runtime)
    should_skip_dir() {
        local d="$1"
        # Direct match or prefix of safe dirs
        for ex in "${safe_prune_excludes[@]}"; do
            if [ "$d" = "$ex" ] || [[ "$d" == $ex/* ]]; then
                return 0
            fi
        done
        # Python virtual environment heuristics
        if [[ "$d" =~ (^|/)(\.venv|venv|env|envs)(/|$) ]]; then
            return 0
        fi
        # site-packages or pycache should remain (they may belong to runtime env)
        if [[ "$d" == *site-packages* ]] || [[ "$d" == *dist-packages* ]]; then
            return 0
        fi
        if [[ "$d" == *__pycache__* ]]; then
            return 0
        fi
        # Skip typical venv subfolders if somehow outside root markers
        case "$d" in
            */bin|*/bin/*|*/lib|*/lib/*|*/include|*/include/*) return 0 ;;
        esac
        return 1
    }

    # Build local directory list (repository tracked tree)
    ( cd "${SCRIPT_DIR}" && find . -type d | sort > "${local_dirs_file}" )
    # Build remote directory list
    ssh -o BatchMode=yes "${NODE_USER}@${NODE_HOST}" "cd '${NODE_CODE_PATH}' && find . -type d | sort" > "${remote_dirs_file}" || { echo "   (warn) could not list remote dirs"; return; }

    # Build prune list = remote minus local
    prune_list=$(comm -23 "${remote_dirs_file}" "${local_dirs_file}")
    if [ -z "${prune_list}" ]; then
        echo "   No directories to prune"
    else
        local removed=0 skipped=0
        while IFS= read -r dir; do
            [ -z "$dir" ] && continue
            if should_skip_dir "$dir"; then
                skipped=$((skipped+1))
                continue
            fi
            echo "   - removing remote directory ${dir}"
            ssh -o BatchMode=yes "${NODE_USER}@${NODE_HOST}" "rm -rf '${NODE_CODE_PATH}/${dir#./}'" || echo "     (warn) failed to remove ${dir}"
            removed=$((removed+1))
        done <<< "${prune_list}"
        echo "   Prune summary: removed=${removed} skipped=${skipped}"
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
    --include='*.png' \
    --include='*.md' \
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
