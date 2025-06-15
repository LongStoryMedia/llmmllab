import os
import sys
import re
from typing import Any, Dict, Optional, Tuple, Set

from util.resolver import SchemaRefResolver
from util.writer import Writer
from util.schema_helpers import to_pascal_case, enum_member_name, enum_member_desc, process_definitions_and_nested_types


class GoGenerator:
    # Common acronyms to preserve in Go code
    COMMON_ACRONYMS = {
        "id": "ID",
        "url": "URL",
        "uri": "URI",
        "api": "API",
        "ui": "UI",
        "uid": "UID",
        "uuid": "UUID",
        "http": "HTTP",
        "https": "HTTPS",
        "html": "HTML",
        "css": "CSS",
        "json": "JSON",
        "xml": "XML",
        "yaml": "YAML",
        "sql": "SQL",
        "db": "DB",
        "ip": "IP",
        "tcp": "TCP",
        "udp": "UDP",
    }

    @staticmethod
    def generate(schema: Dict[str, Any], package_name: str = "main",
                 schema_file: Optional[str] = None,
                 ref_resolver: Optional[SchemaRefResolver] = None,
                 is_main: bool = True,
                 referenced_by: Optional[str] = None,
                 include_yaml_tags: bool = True) -> str:
        """Generate Go types from a JSON schema"""
        # Initialize ref resolver if needed
        if ref_resolver is None and schema_file is not None:
            ref_resolver = SchemaRefResolver(schema_file, schema)

        # Track already processed types to avoid duplicates
        processed_types = set()

        # For external schema files, ensure we generate only the relevant types
        # Flag to indicate if we're generating a standalone file referenced by others
        generating_standalone = not is_main and referenced_by is not None

        # Generate types for the schema
        title = schema.get("title", "").replace(" ", "")
        if not title:
            title = "Root"

        # Find types that should be imported, not redefined
        types_to_import = set()
        if ref_resolver:
            # Find all external schema types that need to be imported
            for ref_path, schema_path in ref_resolver.external_refs.items():
                # Skip self-references
                if schema_file and os.path.abspath(schema_path) == os.path.abspath(schema_file):
                    continue

                # Get the type name from the external schema
                type_name = ref_resolver.external_ref_types.get(
                    schema_path, "")
                if type_name:
                    types_to_import.add(type_name)

        # Create a list of types that will actually be defined in this file
        types_to_generate = set([title])  # Start with the main type

        # Add header comment
        output = [Writer.generate_header_comment("go")]

        # Track needed imports
        needed_imports = set()

        # Pre-analyze file contents to determine what imports we actually need
        if ref_resolver:
            # Analyze the schema to find out which types will be generated in this file
            GoGenerator._analyze_types_in_file(schema, types_to_generate, ref_resolver,
                                               types_to_import, schema_file, is_main, generating_standalone)

            # Now analyze imports but limit to only what's needed for these types
            GoGenerator._analyze_imports_needed(schema, needed_imports, ref_resolver,
                                                types_to_import, types_to_generate)
        else:
            # Without a resolver, just do basic analysis
            GoGenerator._analyze_imports_needed(
                schema, needed_imports, ref_resolver)

        # Add package declaration
        output.append(f"package {package_name}")

        # Only add import section if there are imports
        if needed_imports:
            output.append("\nimport (")
            for imp in sorted(needed_imports):
                output.append(f'\t"{imp}"')
            output.append(")\n")
        else:
            output.append("\n")

        # Track that we're processing this type
        processed_types.add(title)

        def type_callback(sch, _):
            # Extract type name from schema
            type_title = sch.get("title", "").replace(" ", "")

            # Skip types that should be imported, not redefined
            if type_title in types_to_import:
                return ""

            # Skip types that reference external schemas in main file
            if is_main and "$ref" in sch and not sch["$ref"].startswith("#"):
                return ""

            # Skip if this is a definition that maps to an external schema in main file
            if is_main and "title" in sch and ref_resolver and hasattr(ref_resolver, 'definition_to_external_map'):
                if sch["title"] in ref_resolver.definition_to_external_map:
                    return ""

            # For standalone files (referenced from elsewhere)
            if generating_standalone:
                # Only generate this type and its nested types
                if type_title != title and not type_title.startswith(title):
                    return ""

            # Generate the struct
            return GoGenerator._generate_struct_only(sch, schema_file, ref_resolver, processed_types)

        # Generate all nested types
        nested_types = process_definitions_and_nested_types(
            schema, processed_types, ref_resolver, type_callback)
        output.extend(nested_types)

        # Process root struct
        root_struct = GoGenerator._generate_struct_only(
            schema, schema_file, ref_resolver, processed_types)

        # Don't include the root struct if it's in types_to_import
        if title not in types_to_import:
            output.append(root_struct)

        # Special handling for the case of an empty file - ensure we include the type
        if len(output) <= 2:  # Just header, package
            # For standalone files, force generation even with external references
            if generating_standalone and title:
                # Force generating the type without ref resolver
                root_struct_forced = GoGenerator._generate_struct_only(
                    schema, schema_file, None, set())
                if root_struct_forced:
                    output.append(root_struct_forced)

        return "\n".join(output)

    @staticmethod
    def _find_used_external_refs(schema: Dict[str, Any], ref_resolver: SchemaRefResolver) -> Set[str]:
        """Find all external references used in a schema"""
        used_refs = set()

        # Check all properties for external references
        for _, prop_schema in schema.get("properties", {}).items():
            if '$ref' in prop_schema:
                ref_path = prop_schema['$ref']
                # Direct external reference
                if not ref_path.startswith('#'):
                    used_refs.add(ref_path)
                # Reference to a definition that might reference an external schema
                elif ref_path.startswith('#/definitions/'):
                    def_name = ref_path.split('/')[-1]
                    if hasattr(ref_resolver, 'definition_to_external_map') and def_name in ref_resolver.definition_to_external_map:
                        ext_path = ref_resolver.definition_to_external_map[def_name]
                        used_refs.add(ext_path)

        # Check definitions for external references
        if 'definitions' in schema:
            for _, def_schema in schema['definitions'].items():
                if '$ref' in def_schema and not def_schema['$ref'].startswith('#'):
                    used_refs.add(def_schema['$ref'])

        return used_refs

    @staticmethod
    def _analyze_types_in_file(schema: Dict[str, Any], types_in_file: Set[str],
                               ref_resolver: Optional[SchemaRefResolver],
                               types_to_import: Set[str],
                               schema_file: Optional[str] = None,
                               is_main: bool = False,
                               generating_standalone: bool = False,
                               needed_imports: Optional[Set[str]] = None) -> None:
        """
        Analyze schema to determine which types will be defined in this file
        and which will need to be imported
        """
        if needed_imports is None:
            needed_imports = set()
        title = schema.get("title", "").replace(" ", "")
        if not title:
            title = "Root"

        # For the main type and nested types within it, add to types_in_file
        for prop_name, prop_schema in schema.get("properties", {}).items():
            # If property has an enum, it will be defined in this file
            if "enum" in prop_schema:
                enum_type_name = f"{title}{GoGenerator._to_go_field_name(prop_name)}"
                types_in_file.add(enum_type_name)

            # If property is an object with properties, it will be defined in this file
            if prop_schema.get("type") == "object" and "properties" in prop_schema:
                nested_type = "".join(x.capitalize()
                                      for x in prop_name.split("_"))
                types_in_file.add(nested_type)

            # Check if this property references another type
            if "$ref" in prop_schema and ref_resolver:
                ref_path = prop_schema["$ref"]
                # For external references, mark for import
                if not ref_path.startswith('#'):
                    schema_path = os.path.join(os.path.dirname(
                        ref_resolver.base_path), ref_path)
                    if schema_path in ref_resolver.external_ref_types:
                        type_name = ref_resolver.external_ref_types[schema_path]
                        types_to_import.add(type_name)
                # For internal references to definitions in standalone files, include them
                elif ref_path.startswith('#/definitions/') and generating_standalone:
                    def_name = ref_path.split('/')[-1]
                    types_in_file.add(def_name)

        # Also process any definitions that will be included in this file
        if "definitions" in schema and (is_main or generating_standalone):
            for def_name, def_schema in schema["definitions"].items():
                # Skip if this definition references an external schema
                if "$ref" in def_schema and ref_resolver and hasattr(ref_resolver, 'definition_to_external_map') and def_name in ref_resolver.definition_to_external_map:
                    # This definition should be imported, not defined
                    ext_path = ref_resolver.definition_to_external_map[def_name]
                    schema_path = os.path.join(os.path.dirname(
                        ref_resolver.base_path), ext_path)
                    if schema_path in ref_resolver.external_ref_types:
                        type_name = ref_resolver.external_ref_types[schema_path]
                        types_to_import.add(type_name)
                else:
                    # This definition will be defined in this file
                    types_in_file.add(def_name)

                    # If it has properties, analyze them as well
                    if "properties" in def_schema:
                        for _, prop_schema in def_schema.get("properties", {}).items():
                            if prop_schema.get("format") == "uuid":
                                needed_imports.add("github.com/google/uuid")

    @staticmethod
    def _analyze_imports_needed(schema: Dict[str, Any], needed_imports: set,
                                ref_resolver: Optional[SchemaRefResolver] = None,
                                types_to_import: Optional[Set[str]] = None,
                                types_to_generate: Optional[Set[str]] = None):
        """
        Analyze schema to determine which Go imports are needed,
        filtered by the types actually being generated in this file
        """
        # First check if this file needs uuid package
        needs_uuid = False
        needs_time = False
        needs_url = False

        # Helper to check if a property needs special imports
        def check_property_formats(prop_schema):
            nonlocal needs_uuid, needs_time, needs_url

            schema_type = prop_schema.get("type", "")
            schema_format = prop_schema.get("format", "")

            if schema_type == "string":
                if schema_format in ["date-time", "date", "time"]:
                    needs_time = True
                elif schema_format == "uuid":
                    needs_uuid = True
                elif schema_format in ["uri", "url"]:
                    needs_url = True

        # Check properties in the main schema
        for _, prop_schema in schema.get("properties", {}).items():
            check_property_formats(prop_schema)

            # Check array items
            if prop_schema.get("type") == "array" and "items" in prop_schema:
                check_property_formats(prop_schema["items"])

            # Handle references
            if '$ref' in prop_schema and ref_resolver:
                try:
                    resolved_schema = ref_resolver.resolve_ref(
                        prop_schema['$ref'])
                    # No need to check resolved schema here - we're just concerned with formats
                except ValueError:
                    pass

        # Only check definitions if they're going to be defined in this file
        for def_name, def_schema in schema.get("definitions", {}).items():
            # Only check definitions that will be generated in this file
            if types_to_generate is not None and def_name not in types_to_generate:
                continue

            for _, prop_schema in def_schema.get("properties", {}).items():
                check_property_formats(prop_schema)

                # Check array items
                if prop_schema.get("type") == "array" and "items" in prop_schema:
                    check_property_formats(prop_schema["items"])

        # Add needed imports based on what we found
        if needs_uuid:
            needed_imports.add("github.com/google/uuid")
        if needs_time:
            needed_imports.add("time")
        if needs_url:
            needed_imports.add("net/url")

    @staticmethod
    def _generate_struct_only(schema: Dict[str, Any], schema_file: Optional[str] = None,
                              ref_resolver: Optional[SchemaRefResolver] = None,
                              processed_types: Optional[set] = None,
                              parent_title: Optional[str] = None,
                              include_yaml_tags: bool = True) -> str:
        """Generate only the struct definition without package declaration"""
        if processed_types is None:
            processed_types = set()

        title = schema.get("title", "").replace(" ", "")
        if not title:
            title = "Root"

        # Skip if already processed, unless it's the main title itself
        if title in processed_types and title != schema.get("title", ""):
            return ""

        # If this is a standalone file generation, don't skip external refs
        force_generation = ref_resolver is None

        # Check if this type exists as an external reference that should be imported
        if not force_generation and ref_resolver and schema_file and title:
            # Check all external references to see if this type matches any of them
            for ref_path, schema_path in ref_resolver.external_refs.items():
                # Skip this file itself
                if os.path.abspath(schema_path) == os.path.abspath(schema_file):
                    continue

                # Check if the title matches an externally referenced type name
                ext_type_name = ref_resolver.external_ref_types.get(
                    schema_path, "")
                if ext_type_name == title:
                    # Skip generation of this type since it's defined in a separate file
                    return ""

        # Skip if this is a reference to an external schema and not forced
        if not force_generation and "$ref" in schema and ref_resolver:
            ref_path = schema["$ref"]
            if not ref_path.startswith("#"):
                # Don't skip if this is the main type we're generating
                if schema_file and os.path.basename(schema_file) != os.path.basename(ref_path):
                    return ""  # Skip references to other files

        # Skip if this is a definition that maps to an external schema and not forced
        if not force_generation and ref_resolver and hasattr(ref_resolver, 'definition_to_external_map'):
            if title in ref_resolver.definition_to_external_map:
                # Don't skip if this is the main type we're generating
                ext_path = ref_resolver.definition_to_external_map[title]
                if schema_file and os.path.basename(schema_file) != os.path.basename(ext_path):
                    return ""

        processed_types.add(title)
        output = []
        root_desc = schema.get("description", "")

        # --- Collect enums for all properties and generate types/constants ---
        enum_outputs = []
        for prop_name, prop_schema in schema.get("properties", {}).items():
            # If property is an enum, generate a type and constants
            if "enum" in prop_schema:
                enum_type_name = f"{title}{GoGenerator._to_go_field_name(prop_name)}"
                # Avoid duplicate enum type generation
                if processed_types is not None and enum_type_name in processed_types:
                    continue
                if processed_types is not None:
                    processed_types.add(enum_type_name)
                # Patch a schema for the enum type
                enum_schema = prop_schema.copy()
                enum_schema["title"] = enum_type_name
                # Generate the enum type/constants
                enum_outputs.append(GoGenerator._generate_struct_only(
                    enum_schema, schema_file, ref_resolver, processed_types))

        # Handle enum type at the root
        if "enum" in schema:
            enum_values = schema.get("enum", [])
            enum_descriptions = schema.get("enumDescriptions", {})
            enum_names = schema.get("enumNames", {})
            if root_desc:
                output.append(f"// {title} {root_desc}")
            # Always prefix enum constant names with the type name (Go convention)

            def go_enum_const_name(type_name, value, i, enum_names):
                str_value = str(value)
                if isinstance(enum_names, dict) and str_value in enum_names:
                    base = enum_names[str_value]
                elif isinstance(enum_names, list) and i < len(enum_names):
                    base = enum_names[i]
                else:
                    base = to_pascal_case(str(value))
                # Remove type_name prefix if already present
                if base.startswith(type_name):
                    base = base[len(type_name):]
                return f"{type_name}{base}"

            if schema.get("type") == "string":
                output.append(f"type {title} string")
                output.append("")
                output.append("const (")
                for i, value in enumerate(enum_values):
                    const_name = go_enum_const_name(
                        title, value, i, enum_names)
                    desc = enum_member_desc(enum_descriptions, value, i)
                    desc_str = f" // {desc}" if desc else ""
                    output.append(
                        f"\t{const_name} {title} = \"{value}\"{desc_str}")
                output.append(")")
                return "\n".join(output)
            else:
                output.append(f"type {title} int")
                output.append("")
                output.append("const (")
                for i, value in enumerate(enum_values):
                    const_name = go_enum_const_name(
                        title, value, i, enum_names)
                    desc = enum_member_desc(enum_descriptions, value, i)
                    desc_str = f" // {desc}" if desc else ""
                    output.append(
                        f"\t{const_name} {title} = {value}{desc_str}")
                output.append(")")
                return "\n".join(output)

        if root_desc:
            output.append(f"// {title} {root_desc}")

        # Handle root object
        output.append(f"type {title} struct {{")

        # First pass to determine maximum field length for alignment
        field_lengths = []
        for prop_name, _ in schema.get("properties", {}).items():
            field_name = GoGenerator._to_go_field_name(prop_name)
            field_lengths.append(len(field_name))

        max_field_len = max(field_lengths) if field_lengths else 0

        # Second pass to determine maximum type length for alignment
        type_lengths = []
        for prop_name, prop_schema in schema.get("properties", {}).items():
            go_type, _ = GoGenerator._get_go_type(
                prop_schema, prop_name, ref_resolver)
            type_lengths.append(len(go_type))

        max_type_len = max(type_lengths) if type_lengths else 0

        # Now generate the fields with proper alignment
        required = schema.get("required", [])
        for prop_name, prop_schema in schema.get("properties", {}).items():
            field_name = GoGenerator._to_go_field_name(prop_name)
            # If property is an enum, use the generated type
            if "enum" in prop_schema:
                go_type = f"{title}{field_name}"
            else:
                go_type, resolved_schema = GoGenerator._get_go_type(
                    prop_schema, prop_name, ref_resolver)
            is_required = prop_name in required
            if not is_required and not go_type.startswith("[]") and not go_type.startswith("map[") and go_type != "interface{}":
                go_type = f"*{go_type}"

            # Create the combined JSON and YAML tags
            tag_parts = [f'json:"{prop_name}"']
            if include_yaml_tags:
                tag_parts.append(f'yaml:"{prop_name}"')

            combined_tag = f'`{" ".join(tag_parts)}`'

            desc = prop_schema.get("description", "")
            field_padding = " " * (max_field_len - len(field_name))
            type_padding = " " * (max_type_len - len(go_type))

            comment = f"// {desc}" if desc else ""

            if desc:
                output.append(
                    f"\t{field_name}{field_padding} {go_type}{type_padding} {combined_tag} {comment}")
            else:
                output.append(
                    f"\t{field_name}{field_padding} {go_type}{type_padding} {combined_tag}")
        output.append("}\n")

        # Prepend any generated enum types/constants
        return "\n".join(enum_outputs + output)

    @staticmethod
    def _get_go_type(prop_schema: Dict[str, Any], name: str, ref_resolver: Optional[SchemaRefResolver] = None) -> Tuple[str, Dict[str, Any]]:
        """Convert JSON schema type to Go type"""
        # Handle references
        if '$ref' in prop_schema and ref_resolver:
            try:
                resolved_schema = ref_resolver.resolve_ref(prop_schema['$ref'])

                # Get the type name from the reference
                ref_path = prop_schema['$ref']
                type_name = ""

                # For internal references to definitions
                if ref_path.startswith('#/definitions/'):
                    type_name = ref_path.split('/')[-1]
                # For external file references
                elif not ref_path.startswith('#'):
                    # Extract the type name from the filename
                    # e.g., "model_details.json" -> "ModelDetails"
                    filename = os.path.basename(ref_path)
                    base_name = os.path.splitext(
                        filename)[0]  # Remove extension
                    type_name = "".join(x.capitalize()
                                        for x in base_name.split('_'))

                    # Check if this type is already defined in its own file and should be imported
                    # This prevents duplicating types from external references
                    schema_path = os.path.join(os.path.dirname(
                        ref_resolver.base_path), ref_path)
                    if schema_path in ref_resolver.external_ref_types:
                        # Just return the type name without the schema to prevent inlining
                        return type_name, {}

                # If we successfully extracted a type name, use it
                if type_name:
                    return type_name, resolved_schema

                # Otherwise, infer from the property name
                return "".join(x.capitalize() for x in name.split('_')), resolved_schema

            except ValueError as e:
                print(f"Warning: {e}", file=sys.stderr)

        schema_type = prop_schema.get("type", "string")
        schema_format = prop_schema.get("format", "")

        # Handle simple types with formats
        if schema_type == "string":
            if schema_format == "date-time":
                return "time.Time", prop_schema
            elif schema_format == "date":
                return "time.Time", prop_schema
            elif schema_format == "time":
                return "time.Time", prop_schema
            elif schema_format == "duration":
                return "time.Duration", prop_schema
            elif schema_format == "uuid":
                return "uuid.UUID", prop_schema
            elif schema_format == "uri" or schema_format == "url":
                return "url.URL", prop_schema
            else:
                return "string", prop_schema
        elif schema_type == "integer":
            return "int", prop_schema  # Changed from int64 to int
        elif schema_type == "number":
            return "float32", prop_schema  # Changed from float64 to float32
        elif schema_type == "boolean":
            return "bool", prop_schema
        elif schema_type == "array":
            items = prop_schema.get("items", {})
            item_type, _ = GoGenerator._get_go_type(
                items, f"{name}Item", ref_resolver)
            return f"[]{ item_type }", prop_schema
        elif schema_type == "object":
            # For objects, check if they have defined properties
            if "properties" in prop_schema:
                # For nested objects with defined properties, create a new type
                type_name = "".join(x.capitalize() for x in name.split("_"))
                return type_name, prop_schema
            else:
                # For objects without defined properties, use map[string]interface{}
                # Check if additionalProperties is specified
                if "additionalProperties" in prop_schema:
                    add_props = prop_schema["additionalProperties"]
                    if isinstance(add_props, dict) and "type" in add_props:
                        # If additionalProperties specifies a type, use it
                        value_type, _ = GoGenerator._get_go_type(
                            add_props, "value", ref_resolver)
                        return f"map[string]{value_type}", prop_schema
                    elif add_props is True:
                        # If additionalProperties is true, use interface{}
                        return "map[string]interface{}", prop_schema

                # Default to map[string]interface{} for any object type without properties
                return "map[string]interface{}", prop_schema
        else:
            return "interface{}", prop_schema

    @staticmethod
    def _to_go_field_name(name: str) -> str:
        """Convert a property name to a proper Go field name in PascalCase with correct acronym handling."""
        if "_" in name:
            # Handle snake_case names by splitting on underscores and properly capitalizing each part
            words = name.split("_")
            result = []
            for word in words:
                # Check if word is a common acronym
                uppercase_word = GoGenerator.COMMON_ACRONYMS.get(word.lower())
                if uppercase_word:
                    result.append(uppercase_word)
                else:
                    # Otherwise capitalize the first letter
                    result.append(word[0].upper() + word[1:])
            return "".join(result)
        else:
            # Handle camelCase names (already without underscores)
            # First split by capital letters
            words = re.findall(r'[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z]|$)', name)
            result = []
            for word in words:
                # Check if word is a common acronym
                uppercase_word = GoGenerator.COMMON_ACRONYMS.get(word.lower())
                if uppercase_word:
                    result.append(uppercase_word)
                else:
                    # Otherwise capitalize the first letter
                    result.append(word[0].upper() + word[1:])
            return "".join(result)
