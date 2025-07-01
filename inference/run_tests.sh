#!/bin/bash

# Change to the inference directory
cd "$(dirname "$0")" || exit 1

# Install testing dependencies if needed
pip install -r requirements.txt

# Run the tests
python -m pytest test/ -v
