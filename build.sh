#!/bin/bash
#
# Build script for llmmllab
# Generates code from schemas and proto files
#

set -e

SCHEMAS_DIR="./schemas"
PROTO_DIR="./proto"
GEN_DIR="./gen"
INFERENCE_MODELS_DIR="./inference/models"
UI_MODELS_DIR="./ui/src/types"

# Create log file
LOG_FILE="build.log"
echo "Starting build at $(date)" > "$LOG_FILE"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Check if schema2code is available
check_schema2code() {
    if ! command -v schema2code &> /dev/null && ! python -m src.main --help &> /dev/null; then
        log_warn "schema2code not found in PATH. Trying python -m src.main..."
        SCHEMA2CODE_CMD="python -m src.main"
    else
        SCHEMA2CODE_CMD="schema2code"
    fi
    log_info "Using schema2code: $SCHEMA2CODE_CMD"
}

# Generate proto files from schemas
generate_proto_from_schemas() {
    log_info "Generating Protocol Buffer files from schemas..."

    # Get unique proto files referenced in schemas
    local proto_refs=$(grep -rh "import.*\.proto" "$SCHEMAS_DIR" 2>/dev/null | \
        sed 's/.*"\(.*\)".*/\1/' | sort -u)

    # For now, generate from key schemas that map to proto messages
    local schemas=(
        "message"
        "message_content"
        "tool_call"
        "thought"
        "intent_analysis"
        "document"
        "model"
        "model_profile"
        "conversation"
        "dynamic_tool"
        "memory"
        "api_key"
    )

    for schema_name in "${schemas[@]}"; do
        schema_file="$SCHEMAS_DIR/${schema_name}.yaml"
        proto_file="$PROTO_DIR/${schema_name}.proto"

        if [ -f "$schema_file" ]; then
            log_info "Generating $schema_name.proto from $schema_name.yaml"
            $SCHEMA2CODE_CMD "$schema_file" -l proto -o "$proto_file" \
                --package "server" --go-package "github.com/llmmllab/proto"
        else
            log_warn "Schema file not found: $schema_file"
        fi
    done
}

# Generate Python gRPC code from proto files (service-local)
generate_grpc_python() {
    log_info "Generating Python gRPC code from proto files..."

    # Create service-local output directories
    mkdir -p "runner/gen/python"
    mkdir -p "composer/gen/python"
    mkdir -p "server/gen/python"

    # Generate common timestamp for all services
    log_info "Generating common timestamp..."
    python -m grpc_tools.protoc \
        -I "$PROTO_DIR" \
        --python_out="server/gen/python" \
        --grpc_python_out="server/gen/python" \
        "$PROTO_DIR/common/timestamp.proto" 2>&1 || true

    # Generate for runner (uses composer_runner.v1)
    log_info "Generating runner gRPC..."
    python -m grpc_tools.protoc \
        -I "$PROTO_DIR" \
        --python_out="runner/gen/python" \
        --grpc_python_out="runner/gen/python" \
        "$PROTO_DIR/runner/v1/composer_runner.proto" \
        "$PROTO_DIR/common/timestamp.proto" 2>&1 || true

    # Generate for composer (uses both composer_runner.v1 and server_composer.v1)
    log_info "Generating composer gRPC..."
    python -m grpc_tools.protoc \
        -I "$PROTO_DIR" \
        --python_out="composer/gen/python" \
        --grpc_python_out="composer/gen/python" \
        "$PROTO_DIR/composer/v1/server_composer.proto" \
        "$PROTO_DIR/common/timestamp.proto" 2>&1 || true
    # Generate composer_runner.proto (package composer_runner.v1)
    # protoc uses package name to determine subdirs, so output goes to composer_runner/v1/
    python -m grpc_tools.protoc \
        -I "$PROTO_DIR" \
        --python_out="composer/gen/python" \
        --grpc_python_out="composer/gen/python" \
        "$PROTO_DIR/runner/v1/composer_runner.proto" \
        "$PROTO_DIR/common/timestamp.proto" 2>&1 || true

    # Generate for server (uses server_composer.v1)
    # protoc uses package name to determine subdirectory, so it creates composer/v1/
    # We need to fix the generated code to use server_composer.v1 instead of composer.v1
    log_info "Generating server gRPC..."
    mkdir -p "server/gen/python/server_composer/v1/common"
    # Generate timestamp.proto to common/
    python -m grpc_tools.protoc \
        -I "$PROTO_DIR" \
        --python_out="server/gen/python/server_composer/v1" \
        --grpc_python_out="server/gen/python/server_composer/v1" \
        "$PROTO_DIR/common/timestamp.proto" 2>&1 || true
    # Generate server_composer.proto - package is server_composer.v1
    # protoc will create composer/v1/ (last part before .v1)
    python -m grpc_tools.protoc \
        -I "$PROTO_DIR" \
        --python_out="server/gen/python" \
        --grpc_python_out="server/gen/python" \
        "$PROTO_DIR/composer/v1/server_composer.proto" \
        "$PROTO_DIR/common/timestamp.proto" 2>&1 || true
    # Copy generated files from composer/v1/ to server_composer/v1/
    if [ -d "server/gen/python/composer/v1" ]; then
        mkdir -p "server/gen/python/server_composer/v1"
        cp "server/gen/python/composer/v1/"*.py "server/gen/python/server_composer/v1/" 2>/dev/null || true
        rm -rf "server/gen/python/composer"
        # Fix imports in the copied files from composer.v1 to server_composer.v1
        sed -i 's/from composer\.v1 import server_composer_pb2 as composer_dot_v1_dot_server__composer__pb2/from server_composer.v1 import server_composer_pb2 as server_composer_dot_v1_dot_server__composer__pb2/g' "server/gen/python/server_composer/v1/server_composer_pb2_grpc.py"
        sed -i 's/from composer\.v1 import server_composer_pb2 as composer_dot_v1_dot_server__composer__pb2/from server_composer.v1 import server_composer_pb2 as server_composer_dot_v1_dot_server__composer__pb2/g' "server/gen/python/server_composer/v1/server_composer_pb2.py"
        # Fix timestamp import to use relative import
        sed -i 's/from common import timestamp_pb2 as common_dot_timestamp__pb2/from .common import timestamp_pb2 as common_dot_timestamp__pb2/g' "server/gen/python/server_composer/v1/server_composer_pb2.py"
        log_info "Server gRPC generated to server/gen/python/server_composer/v1/"
    fi

    log_info "Python gRPC generation complete"
}

# Generate Python models from schemas
generate_python_models() {
    log_info "Generating Python models from schemas..."

    # List of schemas to skip (missing or external)
    local skip_schemas=(
        "web_search_config"
        "refinement_config"
    )

    # Generate core models
    for schema_file in "$SCHEMAS_DIR"/*.yaml; do
        if [ -f "$schema_file" ]; then
            base_name=$(basename "$schema_file" .yaml)

            # Skip schemas that don't exist or are external
            local should_skip=false
            for skip in "${skip_schemas[@]}"; do
                if [[ "$base_name" == "$skip" ]]; then
                    should_skip=true
                    log_warn "Skipping $base_name.yaml (not available)"
                    break
                fi
            done
            [[ "$should_skip" == "true" ]] && continue

            output_file="$INFERENCE_MODELS_DIR/${base_name}.py"

            # Skip OpenAI schemas (they're API-compatible, not internal)
            if [[ "$base_name" == openai* ]] || [[ "$base_name" == assistant* ]]; then
                continue
            fi

            log_info "Generating $base_name.py"
            $SCHEMA2CODE_CMD "$schema_file" -l python -o "$output_file" 2>&1 || log_warn "Failed to generate $base_name.py"
        fi
    done

    log_info "Python model generation complete"
}

# Generate TypeScript models from schemas
generate_typescript_models() {
    log_info "Generating TypeScript models from schemas..."

    # List of schemas to skip (missing or external)
    local skip_schemas=(
        "web_search_config"
        "refinement_config"
    )

    # Generate core models
    for schema_file in "$SCHEMAS_DIR"/*.yaml; do
        if [ -f "$schema_file" ]; then
            base_name=$(basename "$schema_file" .yaml)

            # Skip schemas that don't exist or are external
            local should_skip=false
            for skip in "${skip_schemas[@]}"; do
                if [[ "$base_name" == "$skip" ]]; then
                    should_skip=true
                    log_warn "Skipping $base_name.yaml (not available)"
                    break
                fi
            done
            [[ "$should_skip" == "true" ]] && continue

            output_file="$UI_MODELS_DIR/${base_name}.ts"

            # Skip OpenAI schemas (they're API-compatible, not internal)
            if [[ "$base_name" == openai* ]] || [[ "$base_name" == assistant* ]]; then
                continue
            fi

            log_info "Generating $base_name.ts"
            $SCHEMA2CODE_CMD "$schema_file" -l typescript -o "$output_file" --package types 2>&1 || log_warn "Failed to generate $base_name.ts"
        fi
    done

    log_info "TypeScript model generation complete"
}

# Update index.ts for TypeScript exports
update_typescript_index() {
    log_info "Updating TypeScript index.ts..."

    local index_file="$UI_MODELS_DIR/index.ts"
    local exports=""

    # Get all exported types
    for ts_file in "$UI_MODELS_DIR"/*.ts; do
        if [ -f "$ts_file" ]; then
            base_name=$(basename "$ts_file" .ts)
            exports+="export * from './${base_name}';"$'\n'
        fi
    done

    echo "$exports" > "$index_file"
    log_info "TypeScript index.ts updated"
}

# Main build process
main() {
    echo "=========================================="
    echo "  llmmllab Build System"
    echo "=========================================="
    echo ""

    check_schema2code

    # Generate proto files from schemas
    generate_proto_from_schemas

    # Generate gRPC code
    generate_grpc_python

    # Generate Python models
    generate_python_models

    # Generate TypeScript models
    generate_typescript_models

    # Update TypeScript index
    update_typescript_index

    echo ""
    echo "=========================================="
    log_info "Build complete! Check $LOG_FILE for details."
    echo "=========================================="
}

# Run main
main "$@"