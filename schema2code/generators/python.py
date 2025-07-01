import os
import sys
from typing import Any, Dict, Optional, List
from util.resolver import SchemaRefResolver
from util.writer import Writer
from util.schema_helpers import to_pascal_case, enum_member_name, enum_member_desc, process_definitions_and_nested_types


class PythonGenerator:
    @staticmethod
    def generate(schema: Dict[str, Any], use_pydantic: bool = True,
                 schema_file: Optional[str] = None,
                 ref_resolver: Optional[SchemaRefResolver] = None,
                 is_main: bool = True, referenced_by: Optional[str] = None) -> str:
        """Generate Python types from a JSON schema"""
        # Initialize ref resolver if needed
        if ref_resolver is None and schema_file is not None:
            ref_resolver = SchemaRefResolver(schema_file, schema)

        # Track already processed types to avoid duplicates
        processed_types = set()

        # Add header comment
        header = [Writer.generate_header_comment("python")]

        # Track required imports based on used types
        imports = [
            "from typing import List, Dict, Optional, Any, Union",
            "from datetime import datetime, date, time, timedelta"
        ]

        # Add imports for external references
        if is_main and ref_resolver and schema_file is not None:
            ext_imports = PythonGenerator._generate_imports(
                ref_resolver, schema_file)
            imports.extend(ext_imports)

        # Add enum import if schema contains an enum
        if "enum" in schema:
            if use_pydantic:
                imports.append("from enum import Enum")
                imports.append(
                    "from pydantic import BaseModel, Field, AnyUrl, EmailStr, constr")
            else:
                imports.append("from enum import Enum")
                imports.append("from dataclasses import dataclass")
        else:
            # Add conditional imports based on used formats
            for _, prop_schema in schema.get("properties", {}).items():
                schema_format = prop_schema.get("format", "")
                if schema_format == "uuid":
                    imports.append("import uuid")
                    break

            if use_pydantic:
                imports.append(
                    "from pydantic import BaseModel, Field, AnyUrl, EmailStr")
            else:
                imports.append("from dataclasses import dataclass")

        output = header + ["\n".join(imports), ""]

        # Process definitions but skip any that are imported from external references
        def type_callback(sch, name):
            # Skip processing if it's an external reference
            if ref_resolver and name in schema.get("definitions", {}):
                def_schema = schema["definitions"][name]
                if "$ref" in def_schema and not def_schema["$ref"].startswith("#"):
                    # This is an external reference, so we don't generate the class
                    return None
            return PythonGenerator._generate_type(sch, use_pydantic, ref_resolver, processed_types)

        definition_outputs = process_definitions_and_nested_types(
            schema, processed_types, ref_resolver, type_callback)

        # Filter out None values (which are skipped external references)
        definition_outputs = [out for out in definition_outputs if out is not None]
        output += definition_outputs

        # Process root type
        root_type = PythonGenerator._generate_type(
            schema, use_pydantic, ref_resolver, processed_types)
        output.append(root_type)

        return "\n\n".join(output)

    @staticmethod
    def _generate_imports(ref_resolver: SchemaRefResolver, current_file: str) -> List[str]:
        """Generate Python import statements for external references"""
        imports = []
        current_dir = os.path.dirname(current_file)

        for ref_path, schema_path in ref_resolver.external_refs.items():
            # Get the type name from the external schema
            type_name = ref_resolver.external_ref_types.get(schema_path, "")
            if not type_name:
                continue

            # Get basename without extension
            basename = os.path.splitext(os.path.basename(ref_path))[0]

            # Add import statement
            imports.append(f"from .{basename} import {type_name}")

        return imports

    @staticmethod
    def _generate_type(schema: Dict[str, Any], use_pydantic: bool,
                       ref_resolver: Optional[SchemaRefResolver] = None,
                       processed_types: Optional[set] = None) -> str:
        """Generate a single Python type from a schema"""
        if processed_types is None:
            processed_types = set()

        title = schema.get("title", "").replace(" ", "")
        if not title:
            title = "Root"

        description = schema.get("description", "")
        required = schema.get("required", [])

        # Handle enum type
        if "enum" in schema:
            # Generate an enum class
            output = []
            enum_values = schema.get("enum", [])
            enum_descriptions = schema.get("enumDescriptions", {})
            enum_names = schema.get("enumNames", {})

            if description:
                output.append(f'"""{description}"""')

            # Use enumNames for member names for both string and integer enums
            if use_pydantic:
                if schema.get("type") == "string":
                    output.append(f"class {title}(str, Enum):")
                else:
                    output.append(f"class {title}(Enum):")
                if not enum_values:
                    output.append("    pass")
                    return "\n".join(output)
                for i, value in enumerate(enum_values):
                    enum_name = enum_member_name(enum_names, value, i, title)
                    desc = enum_member_desc(enum_descriptions, value, i)
                    desc_str = f"  # {desc}" if desc else ""
                    if isinstance(value, str):
                        output.append(f"    {enum_name} = '{value}'{desc_str}")
                    else:
                        output.append(f"    {enum_name} = {value}{desc_str}")
            else:
                output.append(f"class {title}(Enum):")
                if not enum_values:
                    output.append("    pass")
                    return "\n".join(output)
                for i, value in enumerate(enum_values):
                    enum_name = enum_member_name(enum_names, value, i, title)
                    desc = enum_member_desc(enum_descriptions, value, i)
                    desc_str = f"  # {desc}" if desc else ""
                    if isinstance(value, str):
                        output.append(f"    {enum_name} = '{value}'{desc_str}")
                    else:
                        output.append(f"    {enum_name} = {value}{desc_str}")
            return "\n".join(output)

        if use_pydantic:
            output = []
            if description:
                output.append(f'"""{description}"""')
            output.append(f"class {title}(BaseModel):")

            if not schema.get("properties"):
                output.append("    pass")
                return "\n".join(output)

            for prop_name, prop_schema in schema.get("properties", {}).items():
                field_type = PythonGenerator._get_python_type(
                    prop_schema, prop_name, use_pydantic, ref_resolver)

                field_desc = prop_schema.get("description", "")
                is_required = prop_name in required

                if not is_required:
                    field_type = f"Optional[{field_type}] = None"

                if field_desc:
                    field_desc_formatted = field_desc.replace("\n", " ")
                    output.append(f"    # {field_desc_formatted}")

                output.append(f"    {prop_name}: {field_type}")

            # Add Config class
            output.extend([
                "",
                "    class Config:",
                "        extra = \"ignore\""
            ])
        else:
            output = []
            if description:
                output.append(f'"""{description}"""')
            output.append("@dataclass")
            output.append(f"class {title}:")

            if not schema.get("properties"):
                output.append("    pass")
                return "\n".join(output)

            for prop_name, prop_schema in schema.get("properties", {}).items():
                field_type = PythonGenerator._get_python_type(
                    prop_schema, prop_name, use_pydantic)

                field_desc = prop_schema.get("description", "")
                is_required = prop_name in required

                if field_desc:
                    field_desc_formatted = field_desc.replace("\n", " ")
                    output.append(f"    # {field_desc_formatted}")

                if is_required:
                    output.append(f"    {prop_name}: {field_type}")
                else:
                    output.append(
                        f"    {prop_name}: Optional[{field_type}] = None")

        return "\n".join(output)

    @staticmethod
    def _get_python_type(prop_schema: Dict[str, Any], name: str, use_pydantic: bool,
                         ref_resolver: Optional[SchemaRefResolver] = None) -> str:
        """Convert JSON schema type to Python type"""
        # Handle external references specifically
        if '$ref' in prop_schema and ref_resolver:
            ref_path = prop_schema['$ref']
            # If external reference, use the type from the import
            if not ref_path.startswith('#'):
                schema_path = os.path.join(os.path.dirname(
                    ref_resolver.base_path), ref_path)
                if schema_path in ref_resolver.external_ref_types:
                    return ref_resolver.external_ref_types[schema_path]

        # Handle references
        if '$ref' in prop_schema and ref_resolver:
            try:
                resolved_schema = ref_resolver.resolve_ref(prop_schema['$ref'])

                # Extract type name from reference
                ref_path = prop_schema['$ref']
                type_name = ""

                if ref_path.startswith('#/definitions/'):
                    type_name = ref_path.split('/')[-1]
                elif not ref_path.startswith('#'):
                    filename = os.path.basename(ref_path)
                    base_name = os.path.splitext(filename)[0]
                    type_name = "".join(x.capitalize()
                                        for x in base_name.split('_'))

                if type_name:
                    return type_name

                # If unable to extract name, fall back to property name
                return "".join(x.capitalize() for x in name.split('_'))

            except ValueError as e:
                print(f"Warning: {e}", file=sys.stderr)

        schema_type = prop_schema.get("type", "string")
        schema_format = prop_schema.get("format", "")

        if schema_type == "string":
            if schema_format == "date-time":
                return "datetime"
            elif schema_format == "date":
                return "date"
            elif schema_format == "time":
                return "time"
            elif schema_format == "duration":
                return "timedelta"
            elif schema_format == "uuid":
                return "uuid.UUID"
            elif (schema_format == "uri" or schema_format == "url") and use_pydantic:
                return "AnyUrl"
            elif schema_format == "email" and use_pydantic:
                return "EmailStr"
            else:
                return "str"
        elif schema_type == "integer":
            return "int"
        elif schema_type == "number":
            return "float"
        elif schema_type == "boolean":
            return "bool"
        elif schema_type == "array":
            items = prop_schema.get("items", {})

            # Handle references in array items
            if '$ref' in items and ref_resolver:
                ref_path = items['$ref']
                # If external reference, use the type from the import
                if not ref_path.startswith('#'):
                    schema_path = os.path.join(os.path.dirname(
                        ref_resolver.base_path), ref_path)
                    if schema_path in ref_resolver.external_ref_types:
                        return f"List[{ref_resolver.external_ref_types[schema_path]}]"
                elif ref_path.startswith('#/definitions/'):
                    type_name = ref_path.split('/')[-1]
                    return f"List[{type_name}]"

            # If not a reference or couldn't resolve, use the default behavior
            item_type = PythonGenerator._get_python_type(
                items, f"{name}Item", use_pydantic, ref_resolver)
            return f"List[{item_type}]"
        elif schema_type == "object":
            if "properties" in prop_schema:
                # For nested objects with defined properties
                return "".join(x.capitalize() for x in name.split("_"))
            else:
                # For objects without defined properties
                # Check if additionalProperties is specified
                if "additionalProperties" in prop_schema:
                    add_props = prop_schema["additionalProperties"]
                    if isinstance(add_props, dict) and "type" in add_props:
                        # If additionalProperties specifies a type, use it
                        value_type = PythonGenerator._get_python_type(
                            add_props, "value", use_pydantic, ref_resolver)
                        return f"Dict[str, {value_type}]"

                # Default for generic objects
                return "Dict[str, Any]"
        else:
            return "Any"
