#!/bin/bash
# Run the database force initialization script directly
# This can be used for testing or to manually initialize the database

# Set up Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the force init script
echo "Running database force initialization script..."
python db_force_init.py
