#!/bin/bash
# startup.sh - Master script to start all services with the correct virtual environments

# Setup CUDA runtime environment first
bash ./setup_cuda_runtime.sh

# Setup cross-environment access
python3 ./setup_cross_env_access.py

bash ./run.sh

# tail -f /dev/null