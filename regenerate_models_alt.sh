#!/bin/bash

# Set the base directories
SCHEMAS_DIR="./schemas"
MODELS_DIR="./maistro/models"
SCHEMA2CODE="./schema2code/schema2code.py"

# Create a log file
LOG_FILE="regenerate_models.log"
echo "Starting model regeneration at $(date)" >"$LOG_FILE"

# Process each YAML schema file
for schema_file in "$SCHEMAS_DIR"/*.yaml; do
    # Extract the base name without extension
    base_name=$(basename "$schema_file" .yaml)

    # Construct the output Go file path
    go_file="$MODELS_DIR/${base_name}.go"

    echo "Processing schema $base_name" | tee -a "$LOG_FILE"

    # Run schema2code to generate the Go file
    python "$SCHEMA2CODE" "$schema_file" -l go -o "$go_file" --package models

    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully generated $base_name.go" | tee -a "$LOG_FILE"
    else
        echo "Error generating $base_name.go" | tee -a "$LOG_FILE"
    fi
done

echo "Completed model regeneration at $(date)" | tee -a "$LOG_FILE"
