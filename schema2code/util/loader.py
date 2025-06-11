import json
import os
import sys
from typing import Any, Dict
import yaml

YAML_SUPPORT = True


class SchemaLoader:
    """Utility class to load JSON or YAML schema files"""

    def __init__(self, schema_file: str):
        self.schema_file = schema_file
        self.schema = self.load_schema()

    def load_schema(self) -> Dict[str, Any]:
        """Load schema from the specified file"""
        try:
            # Determine file type by extension
            _, ext = os.path.splitext(self.schema_file.lower())

            if ext in ['.yaml', '.yml']:
                # Check if YAML is supported
                if not YAML_SUPPORT:
                    print(
                        "Warning: YAML support requires PyYAML. Install with 'pip install pyyaml'", file=sys.stderr)
                    print("Attempting to parse as JSON instead...", file=sys.stderr)
                    with open(self.schema_file, 'r') as f:
                        return json.load(f)

                if hasattr(yaml, 'safe_load') and yaml is not None:
                    # Parse as YAML
                    with open(self.schema_file, 'r') as f:
                        return yaml.safe_load(f)
                else:
                    # Fallback to JSON if yaml support is incomplete
                    print(
                        "YAML support incomplete. Parsing as JSON instead...", file=sys.stderr)
                    with open(self.schema_file, 'r') as f:
                        return json.load(f)
            else:
                # Default to JSON for all other extensions
                with open(self.schema_file, 'r') as f:
                    return json.load(f)

        except yaml.YAMLError:  # type: ignore
            print(f"Error: {self.schema_file} is not a valid YAML file",
                  file=sys.stderr)
            raise
        except json.JSONDecodeError:
            print(f"Error: {self.schema_file} is not a valid JSON file",
                  file=sys.stderr)
            raise
        except FileNotFoundError:
            print(
                f"Error: Schema file '{self.schema_file}' not found", file=sys.stderr)
            raise
