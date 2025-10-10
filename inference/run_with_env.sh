#!/bin/bash
# run_with_env.sh - Enhanced script for running commands in specific environments

set -e

function v() {
    ./v.sh "$@"
}