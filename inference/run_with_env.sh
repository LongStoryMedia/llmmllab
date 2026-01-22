#!/bin/bash
# run_with_env.sh - Enhanced script for running commands in specific environments

set -e

function v() {
    # Use absolute path since this script may be sourced
    /app/v.sh "$@"
}