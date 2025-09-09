#!/bin/bash
# startup.sh - Master script to start all services with the correct virtual environments

# bash ./setup_environments.sh
python ./set_cross_environment_access.py

bash ./run.sh

# tail -f /dev/null