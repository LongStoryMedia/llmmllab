from typing import Any, Dict
from util.writer import Writer
from util.schema_helpers import to_pascal_case, enum_member_name, enum_member_desc, process_definitions_and_nested_types


class CSharpGenerator:
    @staticmethod
    def generate(schema: Dict[str, Any], namespace: str = "SchemaTypes") -> str:
        """Generate C# types from a JSON schema"""
        # Add header comment
        output = [
            Writer.generate_header_comment("csharp"),
            "using System;",
            "using System.Collections.Generic;",
            "using System.Text.Json.Serialization;",
            "",
            f"namespace {namespace}",
            "{"
        ]

        # Handle enum type in the schema
        if "enum" in schema:
            enum_def = CSharpGenerator._generate_enum(schema, 1)
            output.append(enum_def)
            output.append("}")
            return "\n".join(output)

        processed_types = set()

        def type_callback(sch, _):
            return CSharpGenerator._generate_class(sch, 1)
        output += process_definitions_and_nested_types(
            schema, processed_types, None, type_callback)

        # Process root type
        root_type = CSharpGenerator._generate_class(schema, 1)
        output.append(root_type)

        output.append("}")

        return "\n".join(output)

    @staticmethod
    def _generate_enum(schema: Dict[str, Any], indent_level: int) -> str:
        """Generate a C# enum from a schema with enum values"""
        indent = "    " * indent_level
        title = schema.get("title", "").replace(" ", "")
        if not title:
            title = "Root"

        description = schema.get("description", "")
        output = []

        if description:
            output.append(f"{indent}/// <summary>")
            output.append(f"{indent}/// {description}")
            output.append(f"{indent}/// </summary>")

        # Add JsonConverter attribute for string enums
        if schema.get("type") == "string":
            output.append(
                f"{indent}[JsonConverter(typeof(JsonStringEnumConverter))]")

        output.append(f"{indent}public enum {title}")
        output.append(f"{indent}{{")

        enum_values = schema.get("enum", [])
        enum_descriptions = schema.get("enumDescriptions", {})
        enum_names = schema.get("enumNames", {})

        for i, value in enumerate(enum_values):
            member_name = enum_member_name(enum_names, value, i, title)
            desc = enum_member_desc(enum_descriptions, value, i)
            if desc:
                output.append(f"{indent}    /// <summary>")
                output.append(f"{indent}    /// {desc}")
                output.append(f"{indent}    /// </summary>")
            if schema.get("type") == "string":
                output.append(f'{indent}    [JsonPropertyName("{value}")]')
            comma = "," if i < len(enum_values) - 1 else ""
            if schema.get("type") == "string":
                output.append(f"{indent}    {member_name}{comma}")
            else:
                output.append(f"{indent}    {member_name} = {value}{comma}")

        output.append(f"{indent}}}")
        return "\n".join(output)

    @staticmethod
    def _generate_class(schema: Dict[str, Any], indent_level: int) -> str:
        """Generate a C# class from a schema"""
        indent = "    " * indent_level
        title = schema.get("title", "").replace(" ", "")
        if not title:
            title = "Root"

        description = schema.get("description", "")
        output = []

        if description:
            output.append(f"{indent}/// <summary>")
            output.append(f"{indent}/// {description}")
            output.append(f"{indent}/// </summary>")

        output.append(f"{indent}public class {title}")
        output.append(f"{indent}{{")

        for prop_name, prop_schema in schema.get("properties", {}).items():
            # Convert snake_case to PascalCase for property names
            prop_pascal_case = "".join(x[0].upper() + x[1:]
                                       for x in prop_name.split("_"))

            prop_desc = prop_schema.get("description", "")
            cs_type = CSharpGenerator._get_cs_type(prop_schema, prop_name)

            if prop_desc:
                output.append(f"{indent}    /// <summary>")
                output.append(f"{indent}    /// {prop_desc}")
                output.append(f"{indent}    /// </summary>")

            output.append(f'{indent}    [JsonPropertyName("{prop_name}")]')
            is_required = prop_name in schema.get("required", [])

            if is_required:
                output.append(
                    f"{indent}    public {cs_type} {prop_pascal_case} {{ get; set; }}")
            else:
                if cs_type in ["string", "List<string>", "Dictionary<string, object>"]:
                    output.append(
                        f"{indent}    public {cs_type}? {prop_pascal_case} {{ get; set; }}")
                elif cs_type in ["int", "long", "double", "float", "bool"]:
                    output.append(
                        f"{indent}    public {cs_type}? {prop_pascal_case} {{ get; set; }}")
                else:
                    output.append(
                        f"{indent}    public {cs_type} {prop_pascal_case} {{ get; set; }}")

        output.append(f"{indent}}}")

        return "\n".join(output)

    @staticmethod
    def _get_cs_type(prop_schema: Dict[str, Any], name: str) -> str:
        """Convert JSON schema type to C# type"""
        # Handle enum references
        if "enum" in prop_schema:
            enum_name = "".join(x.capitalize() for x in name.split("_"))
            return enum_name

        schema_type = prop_schema.get("type", "string")
        schema_format = prop_schema.get("format", "")

        if schema_type == "string":
            if schema_format == "date-time":
                return "DateTime"
            elif schema_format == "date":
                return "DateOnly"  # C# 10+ supports DateOnly
            elif schema_format == "time":
                return "TimeOnly"  # C# 10+ supports TimeOnly
            elif schema_format == "duration":
                return "TimeSpan"
            elif schema_format == "uuid":
                return "Guid"
            elif schema_format == "uri" or schema_format == "url":
                return "Uri"
            elif schema_format == "email":
                return "string"
            else:
                return "string"
        elif schema_type == "integer":
            return "long"
        elif schema_type == "number":
            return "double"
        elif schema_type == "boolean":
            return "bool"
        elif schema_type == "array":
            items = prop_schema.get("items", {})
            item_type = CSharpGenerator._get_cs_type(items, f"{name}Item")
            return f"List<{item_type}>"
        elif schema_type == "object":
            if "properties" in prop_schema:
                # For nested objects with defined properties
                return "".join(x[0].upper() + x[1:] for x in name.split("_"))
            else:
                # For generic objects without properties, use Dictionary<string, object>
                # Check if additionalProperties is specified
                if "additionalProperties" in prop_schema:
                    add_props = prop_schema["additionalProperties"]
                    if isinstance(add_props, dict) and "type" in add_props:
                        # If additionalProperties specifies a type, use it
                        value_type = CSharpGenerator._get_cs_type(
                            add_props, "value")
                        return f"Dictionary<string, {value_type}>"

                # Default for objects without defined properties
                return "Dictionary<string, object>"
        else:
            return "object"
