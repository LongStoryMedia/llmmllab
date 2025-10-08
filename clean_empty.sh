#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMPTY_FILES=$(find "$ROOT_DIR" -type f -empty -not -path "$ROOT_DIR/**/venv/*" -not -path "$ROOT_DIR/**/.venv/*" -not -path "$ROOT_DIR/**/node_modules/*" -not -path "$ROOT_DIR/**/__init__.py" -not -path "$ROOT_DIR/.git/*"   )

for file in $EMPTY_FILES; do
    echo "Deleting empty file: $file"
    rm "$file"
done