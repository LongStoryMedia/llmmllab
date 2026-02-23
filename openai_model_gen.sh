#!/bin/bash

# Set the base directories
SCHEMAS_DIR="./schemas/openai"
UI_MODELS_DIR="./ui/src/types/openai"
INFERENCE_MODELS_DIR="./inference/models/openai"

# Create a log file
LOG_FILE="openai_model_gen.log"
echo "Starting model gen at $(date)" >"$LOG_FILE"

rm -f "$SCHEMAS_DIR"/*.py

gen_ts() {
    for schema_file in "$SCHEMAS_DIR"/*.yaml; do
        base_name=$(basename "$schema_file" .yaml)
        # pascal_case_string=$(echo "$base_name" | sed -E 's/(\w+)_?(\w+)/\U$1\U$2/g')
        pascal_case_string=$(echo "$base_name" | sed -E 's/(^|_)([a-z])/\U\2/g' | sed 's/_//g')
        schema2code "$schema_file" -l typescript -o "$UI_MODELS_DIR/${pascal_case_string}.ts" --package types
    done
}

gen_py() {
    for schema_file in "$SCHEMAS_DIR"/*.yaml; do
        base_name=$(basename "$schema_file" .yaml)
        # Convert hyphens and any non-alphanumeric chars (except underscores) to underscores
        module_name=$(echo "$base_name" | sed 's/-/_/g')
        schema2code "$schema_file" -l python -o "$INFERENCE_MODELS_DIR/${module_name}.py"
    done
}

gen_ts
gen_py

wait

echo "Completed openai model generation at $(date)" | tee -a "$LOG_FILE"
