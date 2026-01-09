#!/bin/bash
# Download models from .models.yaml
# Usage: ./download_models.sh [--force]

set -e

FORCE_DOWNLOAD=false
MODELS_YAML="/app/.models.yaml"

# Parse command line arguments
if [[ "$1" == "--force" ]]; then
    FORCE_DOWNLOAD=true
    echo "Force download mode enabled - will re-download existing files"
fi

# Check if .models.yaml exists
if [[ ! -f "$MODELS_YAML" ]]; then
    echo "Error: $MODELS_YAML not found"
    exit 1
fi

echo "Parsing models from $MODELS_YAML..."

# Parse YAML and extract download information
parse_yaml() {
    local current_id=""
    local current_gguf_file=""
    local current_gguf_src=""
    local current_clip_path=""
    local current_mmproj_src=""
    
    while IFS= read -r line; do
        # Skip comments and empty lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue
        
        # New model entry
        if [[ "$line" =~ ^-[[:space:]]+id:[[:space:]]*(.+)$ ]]; then
            # Process previous model if we have data
            if [[ -n "$current_id" && -n "$current_gguf_file" && -n "$current_gguf_src" ]]; then
                echo "MODEL_ID='$current_id'"
                echo "DOWNLOAD_FILE='$current_gguf_file'"
                echo "DOWNLOAD_SRC='$current_gguf_src'"
                echo "DOWNLOAD_TYPE='model'"
                echo "do_download"
                
                if [[ -n "$current_clip_path" && -n "$current_mmproj_src" ]]; then
                    echo "MODEL_ID='$current_id'"
                    echo "DOWNLOAD_FILE='$current_clip_path'"
                    echo "DOWNLOAD_SRC='$current_mmproj_src'"
                    echo "DOWNLOAD_TYPE='mmproj'"
                    echo "do_download"
                fi
            fi
            
            # Start new model
            current_id="${BASH_REMATCH[1]}"
            current_gguf_file=""
            current_gguf_src=""
            current_clip_path=""
            current_mmproj_src=""
        fi
        
        # Extract fields
        if [[ "$line" =~ ^[[:space:]]+gguf_file:[[:space:]]*(.+)$ ]]; then
            current_gguf_file="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ ^[[:space:]]+gguf_src:[[:space:]]*(.+)$ ]]; then
            current_gguf_src="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ ^[[:space:]]+clip_model_path:[[:space:]]*(.+)$ ]]; then
            current_clip_path="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ ^[[:space:]]+mmproj_src:[[:space:]]*(.+)$ ]]; then
            current_mmproj_src="${BASH_REMATCH[1]}"
        fi
    done < "$MODELS_YAML"
    
    # Process last model
    if [[ -n "$current_id" && -n "$current_gguf_file" && -n "$current_gguf_src" ]]; then
        echo "MODEL_ID='$current_id'"
        echo "DOWNLOAD_FILE='$current_gguf_file'"
        echo "DOWNLOAD_SRC='$current_gguf_src'"
        echo "DOWNLOAD_TYPE='model'"
        echo "do_download"
        
        if [[ -n "$current_clip_path" && -n "$current_mmproj_src" ]]; then
            echo "MODEL_ID='$current_id'"
            echo "DOWNLOAD_FILE='$current_clip_path'"
            echo "DOWNLOAD_SRC='$current_mmproj_src'"
            echo "DOWNLOAD_TYPE='mmproj'"
            echo "do_download"
        fi
    fi
}


# Function to perform download
do_download() {
    local dir=$(dirname "$DOWNLOAD_FILE")
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Model: $MODEL_ID ($DOWNLOAD_TYPE)"
    echo "Target: $DOWNLOAD_FILE"
    
    # Check if file exists
    if [[ -f "$DOWNLOAD_FILE" ]] && [[ "$FORCE_DOWNLOAD" == "false" ]]; then
        local size=$(du -h "$DOWNLOAD_FILE" | cut -f1)
        echo "Status: ✓ Already exists ($size) - skipping"
        echo "        Use --force to re-download"
        return 0
    fi
    
    # Create directory if it doesn't exist
    if [[ ! -d "$dir" ]]; then
        echo "Creating directory: $dir"
        mkdir -p "$dir"
    fi
    
    # Download file
    echo "Downloading from: $DOWNLOAD_SRC"
    echo "Progress:"
    
    if curl -L --progress-bar -o "$DOWNLOAD_FILE" "$DOWNLOAD_SRC"; then
        local size=$(du -h "$DOWNLOAD_FILE" | cut -f1)
        echo "Status: ✓ Download complete ($size)"
    else
        echo "Status: ✗ Download failed"
        rm -f "$DOWNLOAD_FILE"  # Clean up partial download
        return 1
    fi
}

# Execute the parser and download commands
eval "$(parse_yaml)"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Download process complete!"
echo ""
