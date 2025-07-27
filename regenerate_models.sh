#!/bin/bash

# Set the base directories
SCHEMAS_DIR="./schemas"
MAISTRO_MODELS_DIR="./maistro/models"
UI_MODELS_DIR="./ui/src/types"
INFERENCE_MODELS_DIR="./inference/models"
PROTO_DIR="./proto"
SCHEMA2CODE="./schema2code/schema2code.py"

# Create a log file
LOG_FILE="regenerate_models.log"
echo "Starting model regeneration at $(date)" >"$LOG_FILE"

# Accept an optional argument for language/project
LANG_ARG=${1:-""}

# Helper function for Go
regen_go() {
    for go_file in "$MAISTRO_MODELS_DIR"/*.go; do
        # Extract the base name without extension
        base_name=$(basename "$go_file" .go)

        # Construct the path to the corresponding schema file
        schema_file="$SCHEMAS_DIR/${base_name}.yaml"

        # Check if schema file exists
        if [ -f "$schema_file" ]; then
            echo "Processing $base_name: Found schema file $schema_file" | tee -a "$LOG_FILE"

            # Run schema2code to regenerate the Go file
            python "$SCHEMA2CODE" "$schema_file" -l go -o "$MAISTRO_MODELS_DIR/${base_name}.go" --package models

            # Check if the command was successful
            if [ $? -eq 0 ]; then
                echo "Successfully regenerated $base_name.go" | tee -a "$LOG_FILE"
            else
                echo "Error regenerating $base_name.go" | tee -a "$LOG_FILE"
            fi
        else
            echo "Skipping $base_name (go): No corresponding schema file found" | tee -a "$LOG_FILE"
        fi
    done
}

# Helper function for TypeScript
regen_ts() {
    for ts_file in "$UI_MODELS_DIR"/*.ts; do
        # Extract the base name without extension
        base_name=$(basename "$ts_file" .ts)
        snake_case_string=$(echo "$base_name" | sed -E 's/([A-Z])/_\1/g' | perl -pe 's/([A-Z])/lc($1)/ge' | sed -E 's/^_//')

        # Construct the path to the corresponding schema file
        schema_file="$SCHEMAS_DIR/${snake_case_string}.yaml"

        # Check if schema file exists
        if [ -f "$schema_file" ]; then
            echo "Processing $base_name: Found schema file $schema_file" | tee -a "$LOG_FILE"

            # Run schema2code to regenerate the TypeScript file
            python "$SCHEMA2CODE" "$schema_file" -l typescript -o "$UI_MODELS_DIR/${base_name}.ts" --package types

            # Check if the command was successful
            if [ $? -eq 0 ]; then
                echo "Successfully regenerated $base_name.ts" | tee -a "$LOG_FILE"
            else
                echo "Error regenerating $base_name.ts" | tee -a "$LOG_FILE"
            fi
        else
            echo "Skipping $base_name (ts): No corresponding schema file found at $schema_file" | tee -a "$LOG_FILE"
        fi
    done
}

# Helper function for Python
regen_py() {
    for py_file in "$INFERENCE_MODELS_DIR"/*.py; do
        # Extract the base name without extension
        base_name=$(basename "$py_file" .py)

        # Construct the path to the corresponding schema file
        schema_file="$SCHEMAS_DIR/${base_name}.yaml"

        # Check if schema file exists
        if [ -f "$schema_file" ]; then
            echo "Processing $base_name: Found schema file $schema_file" | tee -a "$LOG_FILE"

            # Run schema2code to regenerate the Python file
            python "$SCHEMA2CODE" "$schema_file" -l python -o "$INFERENCE_MODELS_DIR/${base_name}.py"

            # Check if the command was successful
            if [ $? -eq 0 ]; then
                echo "Successfully regenerated $base_name.py" | tee -a "$LOG_FILE"
            else
                echo "Error regenerating $base_name.py" | tee -a "$LOG_FILE"
            fi
        else
            echo "Skipping $base_name (py): No corresponding schema file found" | tee -a "$LOG_FILE"
        fi
    done
}

regen_proto() {
    for proto_file in ./proto/*.proto; do
        # Extract the base name without extension
        base_name=$(basename "$proto_file" .proto)

        # Construct the path to the corresponding schema file
        schema_file="$SCHEMAS_DIR/${base_name}.yaml"

        # Check if schema file exists
        if [ -f "$schema_file" ]; then
            echo "Processing $base_name: Found schema file $schema_file" | tee -a "$LOG_FILE"

            # Run schema2code to regenerate the Proto file
            python "$SCHEMA2CODE" "$schema_file" -l proto -o "$PROTO_DIR/${base_name}.proto" --package proto --go-package maistro/proto

            # Check if the command was successful
            if [ $? -eq 0 ]; then
                echo "Successfully regenerated $base_name.proto" | tee -a "$LOG_FILE"
            else
                echo "Error regenerating $base_name.proto" | tee -a "$LOG_FILE"
            fi
        else
            echo "Skipping $base_name (proto): No corresponding schema file found" | tee -a "$LOG_FILE"
        fi
    done
    echo "Generating protobuf code..." | tee -a "$LOG_FILE"
    # python -m grpc_tools.protoc -I proto --python_out=./inference/proto "proto/${base_name}.proto"
    python ./protogen.py --config ./protogen.json

    # echo "Fixing protobuf compatibility issues..." | tee -a "$LOG_FILE"
    # python ./fix_proto_files.py

    # # Verify that the protobuf files are usable
    # echo "Verifying protobuf files..." | tee -a "$LOG_FILE"
    # if python -c "from inference.proto import inference_pb2, inference_pb2_grpc; print('Protobuf imports successful')" 2>/dev/null; then
    #     echo "Protobuf files successfully verified." | tee -a "$LOG_FILE"
    # else
    #     echo "Warning: Protobuf files may have issues. Please check them manually." | tee -a "$LOG_FILE"
    # fi
}

# Always regenerate proto first to ensure all code is in sync
regen_proto

# Then regenerate models for each language
regen_go
regen_ts
regen_py

echo "Completed model regeneration at $(date)" | tee -a "$LOG_FILE"
