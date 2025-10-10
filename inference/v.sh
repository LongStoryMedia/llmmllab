#!/bin/bash

# Default values

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Activate virtual environment and run command
source "/opt/venv/shared/bin/activate"

if [ "$1" = "--interactive" ] || [ "$1" = "" ]; then
    echo -e "${GREEN}Starting interactive shell${NC}"
    echo ""
    echo -e "${YELLOW}Type 'exit' to leave the environment${NC}"
    echo ""
else
    "$@" || {
        exit_code=$?
        echo -e "${RED}Error: Command failed with exit code $exit_code${NC}"
        deactivate
    }
fi