import os
from typing import Any, Dict, Set, Tuple, List

from util.loader import SchemaLoader


class SchemaRefResolver:
    """Helper class to resolve $ref references in JSON schemas"""

    def __init__(self, base_path: str, root_schema: Dict[str, Any]):
        self.base_path = base_path
        self.root_schema = root_schema
        self.loaded_schemas = {}  # Cache for loaded external schemas
        self.external_refs = {}  # Track external schemas and their full paths
        self.external_ref_types = {}  # Map external schema paths to their type names
        self.definition_to_external_map = {}  # Map definitions to external schemas

    def resolve_ref(self, ref_path: str) -> Dict[str, Any]:
        """Resolve a reference path to its schema definition"""
        # Handle internal references to definitions
        if ref_path.startswith('#/definitions/'):
            def_name = ref_path.split('/')[-1]
            if 'definitions' in self.root_schema and def_name in self.root_schema['definitions']:
                ref_value = self.root_schema['definitions'][def_name]
                # Check if this is another reference
                if '$ref' in ref_value:
                    # Store the mapping from definition to external file if applicable
                    if not ref_value['$ref'].startswith('#'):
                        # This is a definition pointing to an external file
                        ext_path = ref_value['$ref']
                        self.definition_to_external_map[def_name] = ext_path
                    return self.resolve_ref(ref_value['$ref'])
                return ref_value
            raise ValueError(
                f"Internal reference '{ref_path}' could not be resolved")

        # Handle external file references
        elif not ref_path.startswith('#'):
            # Load the external schema file
            schema_path = os.path.join(
                os.path.dirname(self.base_path), ref_path)

            if schema_path not in self.loaded_schemas:
                try:
                    loader = SchemaLoader(schema_path)
                    # Load the schema
                    self.loaded_schemas[schema_path] = loader.load_schema()
                    # Track this external reference
                    self.external_refs[ref_path] = schema_path

                    # Create type name from filename
                    filename = os.path.basename(ref_path)
                    base_name = os.path.splitext(filename)[0]
                    type_name = "".join(x.capitalize()
                                        for x in base_name.split('_'))
                    self.external_ref_types[schema_path] = type_name
                    self.external_ref_types[ref_path] = type_name

                except Exception as e:
                    raise ValueError(
                        f"Failed to load external schema '{schema_path}': {e}")

            return self.loaded_schemas[schema_path]

        else:
            raise ValueError(f"Unsupported reference format: {ref_path}")

    def get_external_schemas(self) -> Dict[str, Tuple[Dict[str, Any], str]]:
        """
        Returns a dictionary of external schemas and their file paths
        {schema_path: (schema_data, type_name)}
        """
        result = {}
        for ref_path, schema_path in self.external_refs.items():
            result[schema_path] = (self.loaded_schemas[schema_path],
                                   self.external_ref_types.get(schema_path, ""))
        return result

    def get_type_for_path(self, ref_path: str) -> str:
        """Get the type name for a reference path"""
        # Direct external reference
        if not ref_path.startswith('#'):
            schema_path = os.path.join(
                os.path.dirname(self.base_path), ref_path)
            if ref_path in self.external_ref_types:
                return self.external_ref_types[ref_path]

        # Definition reference that may point to external file
        elif ref_path.startswith('#/definitions/'):
            def_name = ref_path.split('/')[-1]
            if def_name in self.definition_to_external_map:
                ext_path = self.definition_to_external_map[def_name]
                return self.get_type_for_path(ext_path)
            return def_name

        return ""  # Default, should not happen

    def add_external_ref(self, ref_path: str) -> None:
        """
        Add an external reference to the tracked external references
        """
        if not ref_path.startswith('#'):
            schema_path = os.path.join(
                os.path.dirname(self.base_path), ref_path)

            # Add to external references
            self.external_refs[ref_path] = schema_path

            # Try to add to external ref types if not already there
            if schema_path not in self.external_ref_types:
                # Extract type name from filename
                filename = os.path.basename(ref_path)
                base_name = os.path.splitext(filename)[0]
                type_name = "".join(x.capitalize()
                                    for x in base_name.split('_'))

                self.external_ref_types[schema_path] = type_name
