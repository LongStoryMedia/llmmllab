#!/bin/bash
# run_with_env.sh - Enhanced script for running commands in specific environments

set -e

function v() {
    
    # Default values
    ENVIRONMENT=""
    WORKING_DIR="/app"
    
    # Color codes for output
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
    
    usage() {
        echo -e "${BLUE}Usage: v <environment> <command>${NC}"
        echo -e "${BLUE}       v <environment> --interactive${NC}"
        echo ""
        echo -e "${YELLOW}Environments:${NC}"
        echo "  runner     - Use runner virtual environment"
        echo "  server     - Use server virtual environment"
        echo "  evaluation - Use evaluation virtual environment"
        echo ""
        echo -e "${YELLOW}Examples:${NC}"
        echo "  v runner 'python -m runner.main'"
        echo "  v server 'uvicorn server.main:app --host 0.0.0.0 --port 8000'"
        echo "  v evaluation 'python -m evaluation.benchmark'"
        echo "  v runner --interactive  # Interactive shell"
        echo ""
        echo -e "${YELLOW}Cross-environment module access:${NC}"
        echo "  All environments can import from runner, server, and evaluation modules"
        echo "  Example: 'from runner.utils import some_function' works in any environment"
    }
    
    if [ $# -lt 1 ]; then
        echo -e "${RED}Error: Not enough arguments provided${NC}"
        usage
    fi
    
    ENVIRONMENT=$1
    
    
    # Validate environment
    case "$ENVIRONMENT" in
        "runner")
            VENV_PATH="/opt/venv/runner"
        ;;
        "server")
            VENV_PATH="/opt/venv/server"
        ;;
        "evaluation")
            VENV_PATH="/opt/venv/evaluation"
        ;;
        *)
            echo -e "${RED}Error: Invalid environment '$ENVIRONMENT'${NC}"
            usage
            exit 1
        ;;
    esac
    
    shift
    
    # Check if virtual environment exists
    if [ ! -d "$VENV_PATH" ]; then
        echo -e "${RED}Error: Virtual environment '$VENV_PATH' not found${NC}"
        exit 1
    fi
    
    # Set up environment variables for cross-module access
    export PYTHONPATH="/app/runner:/app/server:/app/evaluation:$PYTHONPATH"
    
    echo -e "${GREEN}Activating $ENVIRONMENT environment...${NC}"
    echo -e "${BLUE}Virtual environment: $VENV_PATH${NC}"
    echo -e "${BLUE}Working directory: $WORKING_DIR${NC}"
    echo -e "${BLUE}PYTHONPATH: $PYTHONPATH${NC}"
    echo ""
    
    # Change to working directory
    cd "$WORKING_DIR"
    
    # Activate virtual environment and run command
    source "$VENV_PATH/bin/activate"
    
    if [ "$1" = "--interactive" ] || [ "$1" = "" ]; then
        echo -e "${GREEN}Starting interactive shell in $ENVIRONMENT environment${NC}"
        echo -e "${YELLOW}Cross-environment imports available:${NC}"
        echo "  from runner import ..."
        echo "  from server import ..."
        echo "  from evaluation import ..."
        echo ""
        echo -e "${YELLOW}Type 'exit' to leave the environment${NC}"
        echo ""
    else
        "$@" || {
            local exit_code=$?
            echo -e "${RED}Error: Command failed with exit code $exit_code${NC}"
            deactivate
            cd /app
        }
    fi
}