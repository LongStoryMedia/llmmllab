import os
import sys
from typing import Any, Dict, Optional, Set, List

from util.resolver import SchemaRefResolver
from util.writer import Writer
from util.schema_helpers import enum_member_name, enum_member_desc, process_definitions_and_nested_types, to_pascal_case


class TypeScriptGenerator:
    @staticmethod
    def generate(schema: Dict[str, Any], schema_file: Optional[str] = None,
                 ref_resolver: Optional[SchemaRefResolver] = None,
                 is_main: bool = True, referenced_by: Optional[str] = None) -> str:
        """Generate TypeScript types from a JSON schema"""
        # Initialize ref resolver if needed
        if ref_resolver is None and schema_file is not None:
            ref_resolver = SchemaRefResolver(schema_file, schema)

        # Track already processed types to avoid duplicates
        processed_types = set()

        # Find all external references actually used in this schema
        used_external_refs = set()
        if ref_resolver and schema_file is not None:
            used_external_refs = TypeScriptGenerator._find_used_external_refs(
                schema, ref_resolver)

            # Additional check: explicitly look for external $ref in properties
            # This ensures we don't miss any references
            for _, prop_schema in schema.get("properties", {}).items():
                if '$ref' in prop_schema:
                    ref_path = prop_schema['$ref']
                    if not ref_path.startswith('#'):  # External reference
                        used_external_refs.add(ref_path)
                        # Force the reference to be loaded in the resolver
                        try:
                            ref_resolver.resolve_ref(ref_path)
                        except Exception as e:
                            pass

        # Track imports needed
        imports = []

        # Add imports for external references
        if ref_resolver:
            if schema_file is not None:
                imports = TypeScriptGenerator._generate_imports(
                    ref_resolver, schema_file, used_external_refs)
            else:
                imports = []

        # Add header comment
        output = [Writer.generate_header_comment("typescript")]

        # Add imports if any
        if imports:
            output.extend(imports)
            output.append("")  # Extra line after imports

        # Special handling for TypeScript: don't include definitions that reference external schemas
        # This prevents interface duplication
        def process_ts_defs_callback(schema_obj, processed_set, ref_res, type_cb):
            # Don't process definitions that reference external schemas
            def_outputs = []
            if "definitions" in schema_obj:
                for def_name, def_schema in schema_obj["definitions"].items():
                    if def_name not in processed_set:
                        # Skip definitions that reference external schemas
                        if "$ref" in def_schema and not def_schema["$ref"].startswith("#"):
                            processed_set.add(def_name)
                            continue

                        # Skip definitions that are mapped to external schemas in the resolver
                        if ref_res and hasattr(ref_res, 'definition_to_external_map') and def_name in ref_res.definition_to_external_map:
                            processed_set.add(def_name)
                            continue

                        processed_set.add(def_name)
                        def_schema_copy = def_schema.copy()
                        def_schema_copy["title"] = def_name
                        def_outputs.append(type_cb(def_schema_copy, def_name))

            # Now process nested types in properties
            prop_outputs = []
            for prop_name, prop_schema in schema_obj.get("properties", {}).items():
                # Skip processing external references
                if "$ref" in prop_schema and not prop_schema["$ref"].startswith("#"):
                    continue

                resolved_schema = prop_schema
                type_name = ""
                if "$ref" in prop_schema and ref_res:
                    try:
                        resolved_schema = ref_res.resolve_ref(
                            prop_schema["$ref"])
                        ref_path = prop_schema["$ref"]
                        if ref_path.startswith("#/definitions/"):
                            type_name = ref_path.split("/")[-1]
                        elif not ref_path.startswith("#"):
                            import os
                            filename = os.path.basename(ref_path)
                            base_name = os.path.splitext(filename)[0]
                            type_name = to_pascal_case(base_name)
                        if type_name in processed_set:
                            continue
                    except Exception:
                        pass
                if (resolved_schema.get("type") == "object" and "properties" in resolved_schema) or "properties" in resolved_schema:
                    if not type_name:
                        type_name = to_pascal_case(prop_name)
                    if type_name in processed_set:
                        continue
                    nested_schema = resolved_schema.copy()
                    if "title" not in nested_schema:
                        nested_schema["title"] = type_name
                    processed_set.add(type_name)
                    prop_outputs.append(type_cb(nested_schema, type_name))

            return def_outputs + prop_outputs

        # Custom processing for TypeScript to avoid duplicating interfaces
        types_output = process_ts_defs_callback(schema, processed_types, ref_resolver,
                                                TypeScriptGenerator._generate_interface)
        output.extend(types_output)

        # Process root type
        root_type = TypeScriptGenerator._generate_interface(
            schema, ref_resolver, processed_types)
        output.append(root_type)

        return "\n\n".join(output)

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
    def _generate_imports(ref_resolver: SchemaRefResolver, current_file: str,
                          used_external_refs: Optional[Set[str]] = None) -> List[str]:
        """Generate TypeScript import statements for external references"""
        imports = []
        current_dir = os.path.dirname(current_file)

        # Get the current file's basename to avoid self-imports
        current_basename = os.path.splitext(os.path.basename(current_file))[0]
        current_type_name = "".join(x.capitalize()
                                    for x in current_basename.split('_'))

        # Process explicitly referenced external files
        for ref_path, schema_path in ref_resolver.external_refs.items():
            # Skip if the reference isn't used in this schema
            if used_external_refs is not None and ref_path not in used_external_refs:
                continue

            # Get the type name from the external schema
            type_name = ref_resolver.external_ref_types.get(schema_path, "")
            if not type_name:
                # Try to derive type name from the file path
                ref_basename = os.path.splitext(os.path.basename(ref_path))[0]
                type_name = "".join(x.capitalize()
                                    for x in ref_basename.split('_'))
                if not type_name:
                    continue

            # Skip self-imports (importing from the current file)
            ref_basename = os.path.splitext(os.path.basename(ref_path))[0]
            ref_type_name = "".join(x.capitalize()
                                    for x in ref_basename.split('_'))
            if ref_basename == current_basename or ref_type_name == current_type_name:
                continue

            # Calculate relative path for import
            ref_file = os.path.join(os.path.dirname(current_file), ref_path)
            ref_dir = os.path.dirname(ref_file)

            # Get basename without extension and convert to PascalCase
            basename = "".join(x.capitalize() for x in ref_basename.split('_'))

            # If in same directory, import directly
            if ref_dir == current_dir:
                imports.append(
                    f"import {{ {type_name} }} from './{basename}';")
            else:
                # Calculate relative path
                rel_path = os.path.relpath(ref_dir, current_dir)
                import_path = os.path.join(
                    rel_path, basename).replace('\\', '/')
                if not import_path.startswith('.'):
                    import_path = f"./{import_path}"
                imports.append(
                    f"import {{ {type_name} }} from '{import_path}';")

        return imports

    @staticmethod
    def _generate_interface(schema: Dict[str, Any], ref_resolver: Optional[SchemaRefResolver] = None,
                            processed_types: Optional[set] = None) -> str:
        """Generate a TypeScript interface from a schema"""
        if processed_types is None:
            processed_types = set()

        title = schema.get("title", "").replace(" ", "")
        if not title:
            title = "Root"

        description = schema.get("description", "")
        output = []

        # Handle enum type
        if "enum" in schema:
            enum_values = schema.get("enum", [])
            enum_descriptions = schema.get("enumDescriptions", {})
            enum_names = schema.get("enumNames", {})

            if description:
                output.append("/**")
                output.append(f" * {description}")
                output.append(" */")

            # For string enums, prefer string literal unions (existing logic)
            if schema.get("type") == "string" and all(isinstance(val, str) for val in enum_values):
                # Generate a string literal union type
                type_values = " | ".join([f"'{val}'" for val in enum_values])
                output.append(f"export type {title} = {type_values};")

                # Add a const object with enum values for convenient access
                output.append("")
                output.append("/**")
                output.append(f" * Constant values for {title}")
                output.append(" */")
                output.append(f"export const {title}Values = {{")

                for i, value in enumerate(enum_values):
                    # Get name from enumNames if available
                    const_name = enum_member_name(enum_names, value, i, title)
                    desc = enum_member_desc(enum_descriptions, value, i)
                    desc_str = f" // {desc}" if desc else ""
                    output.append(f"  /** {value}{desc_str} */")
                    output.append(f"  {const_name}: '{value}',")

                output.append("} as const;")
            else:
                # Generate a TypeScript enum (for integer enums and fallback)
                output.append(f"export enum {title} {{")
                for i, value in enumerate(enum_values):
                    enum_name = enum_member_name(enum_names, value, i, title)
                    desc = enum_member_desc(enum_descriptions, value, i)
                    desc_str = f" // {desc}" if desc else ""
                    if isinstance(value, str):
                        output.append(f"  {enum_name} = '{value}',{desc_str}")
                    else:
                        output.append(f"  {enum_name} = {value},{desc_str}")
                output.append("}")
            return "\n".join(output)

        if description:
            output.append("/**")
            output.append(f" * {description}")
            output.append(" */")

        output.append(f"export interface {title} {{")

        for prop_name, prop_schema in schema.get("properties", {}).items():
            prop_desc = prop_schema.get("description", "")
            is_required = prop_name in schema.get("required", [])
            field_suffix = "" if is_required else "?"

            if prop_desc:
                output.append("  /**")
                output.append(f"   * {prop_desc}")
                output.append("   */")

            ts_type = TypeScriptGenerator._get_ts_type(
                prop_schema, prop_name, ref_resolver)
            output.append(f"  {prop_name}{field_suffix}: {ts_type};")

        output.append("}")

        return "\n".join(output)

    @staticmethod
    def _get_ts_type(prop_schema: Dict[str, Any], name: str, ref_resolver: Optional[SchemaRefResolver] = None) -> str:
        """Convert JSON schema type to TypeScript type"""
        # Handle references
        if '$ref' in prop_schema and ref_resolver:
            try:
                ref_path = prop_schema['$ref']

                # First, check definition to external mappings (for definition -> external chains)
                if ref_path.startswith('#/definitions/'):
                    def_name = ref_path.split('/')[-1]
                    if hasattr(ref_resolver, 'definition_to_external_map') and def_name in ref_resolver.definition_to_external_map:
                        # This definition references an external file
                        ext_path = ref_resolver.definition_to_external_map[def_name]
                        schema_path = os.path.join(os.path.dirname(
                            ref_resolver.base_path), ext_path)
                        if ext_path in ref_resolver.external_ref_types:
                            return ref_resolver.external_ref_types[ext_path]
                        # Extract from filename as fallback
                        filename = os.path.basename(ext_path)
                        base_name = os.path.splitext(filename)[0]
                        return "".join(x.capitalize() for x in base_name.split('_'))

                    # Regular definition reference
                    return def_name

                # Handle direct external references
                if not ref_path.startswith('#'):
                    # Get type name from resolver's mapping
                    if hasattr(ref_resolver, 'get_type_for_path'):
                        type_name = ref_resolver.get_type_for_path(ref_path)
                        if type_name:
                            return type_name

                    # Fallback to direct path lookup
                    schema_path = os.path.join(os.path.dirname(
                        ref_resolver.base_path), ref_path)
                    if schema_path in ref_resolver.external_ref_types:
                        return ref_resolver.external_ref_types[schema_path]

                    # Final fallback: extract from filename
                    filename = os.path.basename(ref_path)
                    base_name = os.path.splitext(filename)[0]
                    return "".join(x.capitalize() for x in base_name.split('_'))

                # Try to resolve the reference
                resolved_schema = ref_resolver.resolve_ref(ref_path)

                # If unable to extract name, fall back to property name
                return "".join(x[0].upper() + x[1:] for x in name.split('_'))

            except ValueError as e:
                print(f"Warning: {e}", file=sys.stderr)

        schema_type = prop_schema.get("type", "string")
        schema_format = prop_schema.get("format", "")

        if schema_type == "string":
            # Handle string enum as a union type
            if "enum" in prop_schema:
                enum_values = prop_schema.get("enum", [])
                if all(isinstance(val, str) for val in enum_values):
                    return " | ".join([f"'{val}'" for val in enum_values])

            if schema_format == "date-time":
                return "Date"
            elif schema_format == "date":
                return "Date"
            elif schema_format == "time":
                return "string"  # TypeScript doesn't have a Time type
            elif schema_format == "duration":
                return "string"  # TypeScript doesn't have a Duration type
            elif schema_format == "uuid":
                return "string"  # UUID is generally represented as string
            elif schema_format == "uri" or schema_format == "url":
                return "URL"  # Modern TypeScript can use URL
            elif schema_format == "email":
                return "string"
            else:
                return "string"
        elif schema_type == "integer" or schema_type == "number":
            return "number"
        elif schema_type == "boolean":
            return "boolean"
        elif schema_type == "array":
            items = prop_schema.get("items", {})
            item_type = TypeScriptGenerator._get_ts_type(
                items, f"{name}Item", ref_resolver)
            return f"{item_type}[]"
        elif schema_type == "object":
            if "properties" in prop_schema:
                # For nested objects with defined properties
                return "".join(x[0].upper() + x[1:] for x in name.split("_"))
            else:
                # For generic objects without properties, use Record<string, any>
                # Check if additionalProperties is specified
                if "additionalProperties" in prop_schema:
                    add_props = prop_schema["additionalProperties"]
                    if isinstance(add_props, dict) and "type" in add_props:
                        # If additionalProperties specifies a type, use it
                        value_type = TypeScriptGenerator._get_ts_type(
                            add_props, "value", ref_resolver)
                        return f"Record<string, {value_type}>"

                # Default for objects without defined properties
                return "Record<string, unknown>"
        else:
            return "unknown"
