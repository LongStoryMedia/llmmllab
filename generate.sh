#!/bin/bash

# Make the schema2code script executable
chmod +x "$(dirname "$0")/schema2code/schema2code.py"

FILE=${1:-""}
LALA=${2:-""}

function gen_go() {
    "$(dirname "$0")/schema2code/schema2code.py" "$1" -l go -o "$(dirname "$0")/maistro/models/${2}.go" --package models
}
function gen_python() {
    "$(dirname "$0")/schema2code/schema2code.py" "$1" -l python -o "$(dirname "$0")/inference/models/${2}.py"
}
function gen_typescript() {
    pascal_filename=$(echo "$2" | perl -pe 's/(^|_)([a-z])/uc($2)/ge')
    "$(dirname "$0")/schema2code/schema2code.py" "$1" -l typescript -o "$(dirname "$0")/ui/src/types/${pascal_filename}.ts"
}

function gen() {
    if [ -z "$LALA" ]; then
        gen_go "$1" "$(basename "$1" .yaml)"
        gen_python "$1" "$(basename "$1" .yaml)"
        gen_typescript "$1" "$(basename "$1" .yaml)"
    else
        case "$LALA" in
        go)
            gen_go "$1" "$(basename "$1" .yaml)"
            ;;
        py)
            gen_python "$1" "$(basename "$1" .yaml)"
            ;;
        ts)
            gen_typescript "$1" "$(basename "$1" .yaml)"
            ;;
        *)
            echo "Unsupported language: $LALA"
            exit 1
            ;;
        esac
    fi
}

echo "Starting code generation..."

if [ -z "$FILE" ]; then
    for f in schemas/*.yaml; do
        gen "$f"
    done
else
    gen "schemas/$FILE.yaml"
fi
