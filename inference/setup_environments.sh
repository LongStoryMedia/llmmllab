#!/bin/bash
# setup_environments.sh
# Script to set up all virtual environments for the LLM ML Lab Inference project

set -e  # Exit immediately if a command fails

CURRENT_DIR=$(pwd)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Setting up virtual environments for LLM ML Lab Inference projects..."
echo "---------------------------------------------------------------"

# Function to setup a specific subproject
setup_project() {
  local project_name=$1
  
  echo "Setting up $project_name..."
  cd "$SCRIPT_DIR/$project_name"
  
  # Create virtual environment if it doesn't exist
  if [ ! -d "venv" ]; then
    echo "Creating virtual environment for $project_name..."
    python3.12 -m venv venv
  else
    echo "Virtual environment for $project_name already exists"
  fi
  
  # Activate and install requirements
  echo "Installing requirements for $project_name..."
  source venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  pip install -e .
  deactivate
  
  echo "$project_name setup complete"
  echo "---------------------------------------------------------------"
}

# Setup each project
setup_project "evaluation"
setup_project "server"
setup_project "runner"

# Return to original directory
cd "$CURRENT_DIR"

echo "All environments are set up!"
echo "To activate an environment:"
echo "  cd [project] && source venv/bin/activate"
