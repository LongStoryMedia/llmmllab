# Helper functions for schema2code generators

def to_pascal_case(name: str) -> str:
    """Convert snake_case or lowerCamelCase to PascalCase."""
    if not name:
        return ""
    parts = name.replace('-', '_').split('_')
    return ''.join(x[:1].upper() + x[1:] for x in parts if x)


def to_camel_case(name: str) -> str:
    """Convert snake_case or PascalCase to camelCase."""
    if not name:
        return ""
    # Convert PascalCase to snake_case first
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    parts = s2.split('_')
    return parts[0] + ''.join(x.capitalize() for x in parts[1:] if x)


def enum_member_name(enum_names, value, i, title=None):
    """Get the enum member name from enumNames or fallback."""
    str_value = str(value)
    if isinstance(enum_names, dict) and str_value in enum_names:
        name = enum_names[str_value]
    elif isinstance(enum_names, list) and i < len(enum_names):
        name = enum_names[i]
    else:
        if isinstance(value, int):
            name = f"VALUE_{value}"
        else:
            name = str(value).upper()
    if title and name.startswith(title):
        name = name[len(title):]
    return name


def enum_member_desc(enum_descriptions, value, i):
    """Get the enum member description from enumDescriptions or fallback."""
    str_value = str(value)
    if isinstance(enum_descriptions, dict) and str_value in enum_descriptions:
        return enum_descriptions[str_value]
    elif isinstance(enum_descriptions, list) and i < len(enum_descriptions):
        return enum_descriptions[i]
    return None


def process_definitions_and_nested_types(schema, processed_types, ref_resolver, type_callback):
    """
    Shared utility to process definitions and nested types in a schema.
    - schema: the root schema dict
    - processed_types: set of already processed type names
    - ref_resolver: reference resolver (can be None)
    - type_callback: function(schema_dict, type_name) -> str or None (returns generated type code or None to skip)
    Returns: list of generated type code strings
    """
    output = []
    # Process definitions first
    if "definitions" in schema:
        for def_name, def_schema in schema["definitions"].items():
            if def_name not in processed_types:
                processed_types.add(def_name)

                # Skip if this definition references an external schema (TypeScript special case)
                if "$ref" in def_schema and ref_resolver and hasattr(ref_resolver, 'definition_to_external_map') and def_name in ref_resolver.definition_to_external_map:
                    continue

                # Resolve $ref if present
                if "$ref" in def_schema and ref_resolver:
                    try:
                        resolved_def = ref_resolver.resolve_ref(
                            def_schema["$ref"])
                        def_schema = resolved_def
                    except Exception:
                        pass
                def_schema_copy = def_schema.copy()
                def_schema_copy["title"] = def_name
                type_code = type_callback(def_schema_copy, def_name)
                if type_code is not None:  # Only add non-None returns
                    output.append(type_code)
    # Process nested types in properties
    for prop_name, prop_schema in schema.get("properties", {}).items():
        resolved_schema = prop_schema
        type_name = ""
        if "$ref" in prop_schema and ref_resolver:
            try:
                resolved_schema = ref_resolver.resolve_ref(prop_schema["$ref"])
                ref_path = prop_schema["$ref"]
                if ref_path.startswith("#/definitions/"):
                    type_name = ref_path.split("/")[-1]
                elif not ref_path.startswith("#"):
                    import os
                    filename = os.path.basename(ref_path)
                    base_name = os.path.splitext(filename)[0]
                    type_name = to_pascal_case(base_name)
                if type_name in processed_types:
                    continue
            except Exception:
                pass
        if (resolved_schema.get("type") == "object" and "properties" in resolved_schema) or "properties" in resolved_schema:
            if not type_name:
                type_name = to_pascal_case(prop_name)
            if type_name in processed_types:
                continue
            nested_schema = resolved_schema.copy()
            if "title" not in nested_schema:
                nested_schema["title"] = type_name
            processed_types.add(type_name)
            type_code = type_callback(nested_schema, type_name)
            if type_code is not None:  # Only add non-None returns
                output.append(type_code)
    return output
