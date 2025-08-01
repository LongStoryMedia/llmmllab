#!/bin/bash
# Custom .bashrc file for our Kubernetes pod

# Source the run_with_env.sh to get access to the v() function
if [ -f /app/run_with_env.sh ]; then
    source /app/run_with_env.sh
    # Make the v function available in bash
    # This is needed because functions are not exported by default
    export -f v
fi

# Welcome message
echo -e "\033[1;32m=== Inference Pod Environment ===\033[0m"
echo -e "\033[1;33mAvailable environments:\033[0m"
echo "  - runner     (use: v runner [command])"
echo "  - server     (use: v server [command])"
echo "  - evaluation (use: v evaluation [command])"
echo ""
echo -e "\033[1;33mExample commands:\033[0m"
echo "  v runner python -m runner.main"
echo "  v server 'uvicorn server.main:app --host 0.0.0.0 --port 8000'"
echo ""
exec /bin/bash