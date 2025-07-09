#!/usr/bin/env python3
"""
schema2code - A tool for converting JSON schemas to programming language types
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from generators.dotnet import CSharpGenerator
from generators.go import GoGenerator
from generators.python import PythonGenerator
from generators.typescript import TypeScriptGenerator
from generators.proto import ProtoGenerator
from util.writer import Writer
from util.loader import SchemaLoader
from util.resolver import SchemaRefResolver


def generate_types(schema_data: Dict, language: str, schema_file: str, **kwargs) -> Dict[str, str]:
    """
    Generate types for the specified language
    Returns a dictionary mapping output files to their content
    """
    # Create ref resolver
    ref_resolver = SchemaRefResolver(schema_file, schema_data)

    # Dictionary to store output files and their content
    output_files = {}

    # Pre-process to resolve all references, especially chained ones
    _preprocess_schema_references(ref_resolver, schema_data)

    # Generate the main schema types
    main_output = _generate_single_schema(
        schema_data, language, schema_file, ref_resolver, is_main=True, **kwargs)

    output_file = kwargs.get('output', 'output')
    output_files[output_file] = main_output

    # Get external schemas that need their own files
    external_schemas = ref_resolver.get_external_schemas()

    # Generate types for each external schema
    for schema_path, (ext_schema_data, type_name) in external_schemas.items():
        # Generate a filename for this external schema
        filename = os.path.basename(schema_path)
        basename, _ = os.path.splitext(filename)

        # Get output directory
        output_dir = os.path.dirname(output_file)

        # Create output filename based on the language
        if language == "typescript":
            output_ext = ".ts"
            # For TypeScript, use PascalCase filenames
            basename = "".join(x.capitalize() for x in basename.split('_'))
        elif language == "python":
            output_ext = ".py"
        elif language == "go":
            output_ext = ".go"
        elif language == "csharp" or language == "dotnet":
            output_ext = ".cs"
        elif language == "proto" or language == "protobuf":
            output_ext = ".proto"
        else:
            output_ext = ".txt"

        ext_output_file = os.path.join(output_dir, f"{basename}{output_ext}")

        # Generate types for this external schema
        ext_output = _generate_single_schema(
            ext_schema_data, language, schema_path, ref_resolver,
            is_main=False, referenced_by=output_file, type_name=type_name, **kwargs)

        # Add to output files
        output_files[ext_output_file] = ext_output

    return output_files


def _generate_single_schema(schema_data: Dict, language: str, schema_file: str,
                            ref_resolver: SchemaRefResolver, is_main: bool = True,
                            referenced_by: Optional[str] = None, type_name: Optional[str] = None, **kwargs) -> str:
    """Generate types for a single schema file"""
    if language == "go":
        package_name = kwargs.get("package_name", "main")
        return GoGenerator.generate(schema_data, package_name, schema_file, ref_resolver,
                                    is_main=is_main, referenced_by=referenced_by)
    elif language == "python":
        use_pydantic = kwargs.get("use_pydantic", True)
        return PythonGenerator.generate(schema_data, use_pydantic, schema_file, ref_resolver,
                                        is_main=is_main, referenced_by=referenced_by)
    elif language == "typescript":
        return TypeScriptGenerator.generate(schema_data, schema_file, ref_resolver,
                                            is_main=is_main, referenced_by=referenced_by)
    elif language == "csharp" or language == "dotnet":
        namespace = kwargs.get("namespace", "SchemaTypes")
        return CSharpGenerator.generate(schema_data, namespace)
    elif language == "proto" or language == "protobuf":
        package_name = kwargs.get("package_name", "schema")
        go_package = kwargs.get("go_package")
        return ProtoGenerator.generate(schema_data, package_name, schema_file, ref_resolver,
                                       is_main=is_main, referenced_by=referenced_by, go_package=go_package)
    else:
        raise ValueError(f"Unsupported language: {language}")


def _preprocess_schema_references(ref_resolver: SchemaRefResolver, schema: Dict[str, Any]) -> None:
    """
    Pre-process all references in the schema to resolve chained references and prepare external refs
    """
    # Process all property references
    for prop_name, prop_schema in schema.get("properties", {}).items():
        if "$ref" in prop_schema:
            ref_path = prop_schema["$ref"]

            # For external references, add to the used_refs
            if not ref_path.startswith("#"):
                ref_resolver.add_external_ref(ref_path)

            try:
                # Resolve the reference to populate resolver's cache
                resolved_schema = ref_resolver.resolve_ref(ref_path)

                # Also process nested references in the resolved schema
                if isinstance(resolved_schema, dict):
                    _preprocess_schema_references(ref_resolver, resolved_schema)
            except Exception as e:
                print(
                    f"Warning: Could not resolve reference {ref_path}: {e}", file=sys.stderr)

    # Also process any definitions
    if "definitions" in schema:
        for def_name, def_schema in schema["definitions"].items():
            if "$ref" in def_schema:
                ref_path = def_schema["$ref"]

                # For external references, add to the used_refs
                if not ref_path.startswith("#"):
                    ref_resolver.add_external_ref(ref_path)

                try:
                    # Resolve the reference
                    resolved_schema = ref_resolver.resolve_ref(ref_path)

                    # Also process nested references in the resolved schema
                    if isinstance(resolved_schema, dict):
                        _preprocess_schema_references(ref_resolver, resolved_schema)
                except Exception as e:
                    print(
                        f"Warning: Could not resolve definition reference {ref_path}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Generate code types from JSON or YAML schema")
    parser.add_argument(
        "schema_file", help="Path to the JSON or YAML schema file")
    parser.add_argument(
        "--language", "-l",
        choices=["go", "python", "typescript", "csharp", "dotnet", "proto", "protobuf"],
        required=True,
        help="Target language for code generation"
    )
    parser.add_argument("--output", "-o", required=True,
                        help="Output file path")
    parser.add_argument(
        "--mode",
        choices=["create", "append"],
        default="create",
        help="Whether to create a new file or append to existing"
    )
    parser.add_argument(
        "--no-create",
        action="store_true",
        help="Don't create the file if it doesn't exist"
    )
    # Add a new flag to prevent overwriting existing files
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Don't overwrite the file if it already exists"
    )

    # Language-specific options
    parser.add_argument("--package", default="main", help="Go package name or Protocol Buffer package name")
    parser.add_argument(
        "--namespace", default="SchemaTypes", help="C# namespace")
    parser.add_argument("--no-pydantic", action="store_true",
                        help="Use dataclasses instead of Pydantic for Python")
    parser.add_argument("--go-package", help="Go package option for Protocol Buffer files")

    args = parser.parse_args()

    try:
        loader = SchemaLoader(args.schema_file)
        schema_data = loader.load_schema()

        output_files = generate_types(
            schema_data,
            args.language,
            args.schema_file,
            package_name=args.package,
            namespace=args.namespace,
            use_pydantic=not args.no_pydantic,
            go_package=args.go_package,
            output=args.output
        )

        # Write all output files
        success = Writer.write_multiple_files(
            output_files,
            args.mode,
            not args.no_create,
            args.no_overwrite
        )

        if success:
            print(
                f"Successfully generated {args.language} types in {len(output_files)} file(s)")
            return 0
        else:
            return 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
