#!/bin/bash
# Script to check the dataset structure

set -x  # Print commands as they are executed

# Check working directory and Python environment
echo "Current directory: $(pwd)"
echo "Python version: $(python --version 2>&1)"

# Default dataset path
DATASET_PATH="./evaluations/datasets/samples/multiple_choice_dataset.json"
if [ ! -z "$1" ]; then
  DATASET_PATH="$1"
fi

echo "Checking dataset at: $DATASET_PATH"

# Check if dataset file exists
if [ -f "$DATASET_PATH" ]; then
    echo "Dataset file exists"
    echo "File size: $(du -h $DATASET_PATH | cut -f1)"
    
    # Print file contents
    echo "Dataset contents:"
    cat "$DATASET_PATH"
    
    # Validate JSON format
    echo "Validating JSON format..."
    python -c "import json; print(json.load(open('$DATASET_PATH')))" && echo "Valid JSON" || echo "INVALID JSON"
    
    # Check dataset structure
    echo "Dataset structure check:"
    python -c "
import json, sys
try:
    data = json.load(open('$DATASET_PATH'))
    print('Dataset keys:', list(data.keys()))
    
    # Check for metadata
    if 'metadata' in data:
        print('Metadata:', data['metadata'])
    else:
        print('WARNING: No metadata found')
    
    # Check for examples or direct lists
    if 'examples' in data:
        print('Examples count:', len(data['examples']))
        print('First example keys:', list(data['examples'][0].keys()) if data['examples'] else 'None')
    elif all(k in data for k in ['questions', 'choices', 'answers']):
        print('Direct format - questions count:', len(data['questions']))
    elif all(k in data for k in ['prompts', 'references']):
        print('Text generation format - prompts count:', len(data['prompts']))
    else:
        print('WARNING: Unknown dataset structure')
        print('Available keys:', list(data.keys()))
except Exception as e:
    print('ERROR:', str(e))
    sys.exit(1)
"
else
    echo "Dataset file NOT found!"
    echo "Searching for any JSON files..."
    find . -name "*.json" | grep -i -E "dataset|sample"
fi

# Check the evaluation script
EVAL_SCRIPT="./evaluations/run_model_eval.py"
if [ -f "$EVAL_SCRIPT" ]; then
    echo "Evaluation script found"
    echo "Checking dataset loading function:"
    grep -A 20 "def load_dataset" "$EVAL_SCRIPT"
else
    echo "Evaluation script NOT found at $EVAL_SCRIPT!"
fi
