#!/bin/bash

# Change to the inference directory
cd "$(dirname "$0")" || exit 1

# Install testing dependencies if needed (server layer)
pip install -r server/requirements.txt >/dev/null 2>&1 || true

# Run the tests
# Run only server tests for faster feedback if available, else fallback
if [ -d "server/test" ]; then
	python -m pytest server/test -q
else
	python -m pytest -q
fi
