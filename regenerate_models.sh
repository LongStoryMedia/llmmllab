#!/bin/bash

# Set the base directories
SCHEMAS_DIR="./schemas"
MAISTRO_MODELS_DIR="./maistro/models"
UI_MODELS_DIR="./ui/src/types"
COMPOSER_MODELS_DIR="./composer/models"
RUNNER_MODELS_DIR="./runner/models"
SERVER_MODELS_DIR="./server/models"
PROTO_DIR="./proto"

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
            schema2code "$schema_file" -l go -o "$MAISTRO_MODELS_DIR/${base_name}.go" --package models
            
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

tsgen() {
    schema_file=$1
    
    # Run schema2code to regenerate the TypeScript file
    schema2code "$schema_file" -l typescript -o "$UI_MODELS_DIR/${base_name}.ts" --package types
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully regenerated $schema_file in typescript" | tee -a "$LOG_FILE"
    else
        echo "Error regenerating $schema_file in typescript" | tee -a "$LOG_FILE"
    fi
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
            
            tsgen "$schema_file" &
        else
            echo "Skipping $base_name (ts): No corresponding schema file found at $schema_file" | tee -a "$LOG_FILE"
        fi
    done
}

pygen() {
    schema_file=$1
    output_dir=$2
    base_name=$3
    
    # Run schema2code to regenerate the Python file
    schema2code "$schema_file" -l python -o "${output_dir}/${base_name}.py"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully regenerated ${base_name}.py in ${output_dir}" | tee -a "$LOG_FILE"
    else
        echo "Error regenerating ${base_name}.py" | tee -a "$LOG_FILE"
    fi
}

# Helper function for Python - Composer models
regen_py_composer() {
    for py_file in "$COMPOSER_MODELS_DIR"/*.py; do
        # Extract the base name without extension
        base_name=$(basename "$py_file" .py)
        
        # Construct the path to the corresponding schema file
        schema_file="$SCHEMAS_DIR/${base_name}.yaml"
        
        # Check if schema file exists
        if [ -f "$schema_file" ]; then
            echo "Processing $base_name (composer): Found schema file $schema_file" | tee -a "$LOG_FILE"
            pygen "$schema_file" "$COMPOSER_MODELS_DIR" "$base_name" &
        else
            echo "Skipping $base_name (composer): No corresponding schema file found" | tee -a "$LOG_FILE"
        fi
    done
}

# Helper function for Python - Runner models
regen_py_runner() {
    for py_file in "$RUNNER_MODELS_DIR"/*.py; do
        # Extract the base name without extension
        base_name=$(basename "$py_file" .py)
        
        # Construct the path to the corresponding schema file
        schema_file="$SCHEMAS_DIR/${base_name}.yaml"
        
        # Check if schema file exists
        if [ -f "$schema_file" ]; then
            echo "Processing $base_name (runner): Found schema file $schema_file" | tee -a "$LOG_FILE"
            pygen "$schema_file" "$RUNNER_MODELS_DIR" "$base_name" &
        else
            echo "Skipping $base_name (runner): No corresponding schema file found" | tee -a "$LOG_FILE"
        fi
    done
}

# Helper function for Python - Server models
regen_py_server() {
    for py_file in "$SERVER_MODELS_DIR"/*.py; do
        # Extract the base name without extension
        base_name=$(basename "$py_file" .py)
        
        # Construct the path to the corresponding schema file
        schema_file="$SCHEMAS_DIR/${base_name}.yaml"
        
        # Check if schema file exists
        if [ -f "$schema_file" ]; then
            echo "Processing $base_name (server): Found schema file $schema_file" | tee -a "$LOG_FILE"
            pygen "$schema_file" "$SERVER_MODELS_DIR" "$base_name" &
        else
            echo "Skipping $base_name (server): No corresponding schema file found" | tee -a "$LOG_FILE"
        fi
    done
}

# Helper function for Python - Legacy inference models (for backward compatibility)
regen_py_inference() {
    for py_file in "$INFERENCE_MODELS_DIR"/*.py; do
        # Extract the base name without extension
        base_name=$(basename "$py_file" .py)
        
        # Construct the path to the corresponding schema file
        schema_file="$SCHEMAS_DIR/${base_name}.yaml"
        
        # Check if schema file exists
        if [ -f "$schema_file" ]; then
            echo "Processing $base_name (inference): Found schema file $schema_file" | tee -a "$LOG_FILE"
            pygen "$schema_file" "$INFERENCE_MODELS_DIR" "$base_name" &
        else
            echo "Skipping $base_name (inference): No corresponding schema file found" | tee -a "$LOG_FILE"
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
            schema2code "$schema_file" -l proto -o "$PROTO_DIR/${base_name}.proto" --package proto --go-package maistro/proto
            
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
    # if python -c "from proto import inference_pb2, inference_pb2_grpc; print('Protobuf imports successful')" 2>/dev/null; then
    #     echo "Protobuf files successfully verified." | tee -a "$LOG_FILE"
    # else
    #     echo "Warning: Protobuf files may have issues. Please check them manually." | tee -a "$LOG_FILE"
    # fi
}

# Always regenerate proto first to ensure all code is in sync
# regen_proto

# Then regenerate models for each language
regen_ts
regen_py_composer
regen_py_runner
regen_py_server
# regen_py_inference

wait

echo "Completed model regeneration at $(date)" | tee -a "$LOG_FILE"
