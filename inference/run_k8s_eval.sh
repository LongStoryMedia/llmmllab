#!/bin/bash
# Script to run evaluations in Kubernetes environment
set -e  # Exit on any error
set -x  # Print commands as they are executed (debug mode)

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Run model evaluations in Kubernetes environment"
    echo 
    echo "Options:"
    echo "  -m, --model-id MODEL_ID    Specify the model ID to evaluate (default: qwen3-30b-a3b-q4-k-m)"
    echo "  -t, --task TASK_TYPE       Specify the task type: text-gen, multiple_choice, mc, qa (default: multiple_choice)"
    echo "  -d, --dataset PATH         Path to dataset JSON file (default: ./evaluations/datasets/samples/multiple_choice_dataset.json)"
    echo "  -o, --output-dir DIR       Directory to save results (default: ./results)"
    echo "  -h, --help                 Show this help message"
    exit 1
}

# Default values
MODEL_ID="qwen3-30b-a3b-q4-k-m"
TASK="text-gen"
DATASET="./evaluations/datasets/samples/text_generation_test.json"
OUTPUT_DIR="./results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -m|--model-id)
            MODEL_ID="$2"
            shift 2
            ;;
        -t|--task)
            TASK="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Environment checks
echo "Running in directory: $(pwd)"
echo "Python version: $(python --version 2>&1)"
echo "Python path: $(which python 2>&1)"
echo "User: $(whoami)"

echo "Current directory contents:"
ls -la

echo "Dataset file exists check:"
if [ -f "$DATASET" ]; then
    echo "Dataset exists: $DATASET"
    echo "File size: $(du -h $DATASET | cut -f1)"
    echo "File contents preview:"
    head -n 20 "$DATASET"
else
    echo "Dataset does NOT exist: $DATASET"
    echo "Checking for any dataset files:"
    find . -name "*.json" | grep -i dataset
fi

echo "Running evaluation for model: $MODEL_ID"
echo "Task type: $TASK"
echo "Dataset: $DATASET"
echo "Output directory: $OUTPUT_DIR"

# Make sure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Execute the evaluation command with explicit Python interpreter
echo "Executing evaluation command..."
python -m evaluations.run_model_eval --model-id "$MODEL_ID" --task "$TASK" --dataset "$DATASET" --output-dir "$OUTPUT_DIR" --verbose

exit_code=$?
echo "Command exited with code: $exit_code"

if [ $exit_code -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo "Results should be saved to: $OUTPUT_DIR/$MODEL_ID/"
    echo "Output directory contents:"
    ls -la "$OUTPUT_DIR"
    if [ -d "$OUTPUT_DIR/$MODEL_ID/" ]; then
        echo "Model output directory contents:"
        ls -la "$OUTPUT_DIR/$MODEL_ID/"
    fi
else
    echo "Evaluation failed with exit code $exit_code"
    echo "Checking Python module existence:"
    python -c "import sys; print(sys.path)"
    python -c "import evaluations; print('evaluations module found')" || echo "evaluations module not found"
fi
