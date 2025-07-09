#!/bin/bash
# Example script to demonstrate the Protocol Buffer generator

# Define paths
SCHEMA_PATH="./examples/person.json"
OUTPUT_PATH="./examples/person.proto"

# Check if the example schema exists, if not create it
if [ ! -f "$SCHEMA_PATH" ]; then
    mkdir -p "$(dirname "$SCHEMA_PATH")"
    cat >"$SCHEMA_PATH" <<'EOF'
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Person",
  "type": "object",
  "properties": {
    "id": {
      "type": "integer",
      "description": "The unique identifier for a person"
    },
    "name": {
      "type": "string",
      "description": "The person's full name"
    },
    "email": {
      "type": "string",
      "format": "email",
      "description": "Email address"
    },
    "birthDate": {
      "type": "string",
      "format": "date-time",
      "description": "Date of birth with time"
    },
    "address": {
      "type": "object",
      "description": "The person's address",
      "properties": {
        "street": {
          "type": "string"
        },
        "city": {
          "type": "string"
        },
        "state": {
          "type": "string"
        },
        "postalCode": {
          "type": "string"
        },
        "country": {
          "type": "string"
        }
      }
    },
    "phoneNumbers": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string",
            "enum": ["home", "work", "mobile"],
            "description": "Type of phone number"
          },
          "number": {
            "type": "string",
            "description": "Phone number"
          }
        }
      }
    },
    "status": {
      "type": "string",
      "enum": ["active", "inactive", "pending"],
      "description": "Account status"
    },
    "metadata": {
      "type": "object",
      "additionalProperties": true,
      "description": "Additional metadata as key-value pairs"
    }
  },
  "required": ["id", "name", "email"],
  "additionalProperties": false
}
EOF
    echo "Created example schema at $SCHEMA_PATH"
fi

# Generate the Protocol Buffer file
echo "Generating Protocol Buffer file..."
python3 schema2code.py "$SCHEMA_PATH" --language proto --output "$OUTPUT_PATH" --package "personschema" --go-package "example/person"

# Show the result if successful
if [ $? -eq 0 ]; then
    echo "Successfully generated Protocol Buffer file at $OUTPUT_PATH"
    echo "Generated content:"
    echo "===================="
    cat "$OUTPUT_PATH"
    echo "===================="
else
    echo "Failed to generate Protocol Buffer file"
fi
