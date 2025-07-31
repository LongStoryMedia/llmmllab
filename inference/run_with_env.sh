#!/bin/bash
# run_with_env.sh - Helper script to run commands within a specific virtual environment
# Usage: ./run_with_env.sh <env_name> <command> [args...]
# Example: ./run_with_env.sh server python -m uvicorn app:app --port 8000

set -e

# Color codes for better visibility
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if an environment name was provided
if [ $# -lt 2 ]; then
    echo -e "${RED}Error: Please provide an environment name and command${NC}"
    echo -e "Usage: ${YELLOW}$0 <env_name> <command> [args...]${NC}"
    echo -e "Available environments: ${GREEN}evaluation${NC}, ${GREEN}server${NC}, ${GREEN}runner${NC}"
    echo -e "Example: ${BLUE}$0 server python -m uvicorn app:app --port 8000${NC}"
    exit 1
fi

# Extract the environment name from arguments
ENV_NAME=$1
shift

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo -e "${BLUE}Using script directory: ${YELLOW}$SCRIPT_DIR${NC}"

# Map environment name to its path
case "$ENV_NAME" in
    evaluation)
        VENV_PATH="${EVALUATION_VENV:-$SCRIPT_DIR/evaluation/venv}"
        DIR_PATH="$SCRIPT_DIR/evaluation"
        ;;
    server)
        VENV_PATH="${SERVER_VENV:-$SCRIPT_DIR/server/venv}"
        DIR_PATH="$SCRIPT_DIR/server"
        ;;
    runner)
        VENV_PATH="${RUNNER_VENV:-$SCRIPT_DIR/runner/venv}"
        DIR_PATH="$SCRIPT_DIR/runner"
        ;;
    *)
        echo -e "${RED}Error: Unknown environment '$ENV_NAME'${NC}"
        echo -e "Available environments: ${GREEN}evaluation${NC}, ${GREEN}server${NC}, ${GREEN}runner${NC}"
        exit 1
        ;;
esac

# Check if directory exists
if [ ! -d "$DIR_PATH" ]; then
    echo -e "${RED}Error: Directory not found at $DIR_PATH${NC}"
    exit 1
fi

# Execute the command in the specified environment
echo -e "${BLUE}Running in ${GREEN}$ENV_NAME${BLUE} environment: ${YELLOW}$*${NC}"
echo -e "${BLUE}Working directory: ${YELLOW}$DIR_PATH${NC}"

# Change to the directory
cd "$DIR_PATH" || { 
    echo -e "${RED}Failed to change to $DIR_PATH${NC}"
    exit 1
}

# Activate the virtual environment and run the command
source "$VENV_PATH/bin/activate" || {
    echo -e "${RED}Failed to activate $VENV_PATH${NC}"
    exit 1
}

# Execute the command with exec to replace the current process
exec "$@"
