#!/usr/bin/env python3
"""
Protocol buffer code generator.

This script generates Python gRPC code from .proto files defined in protogen.json.
It handles package-specific output directories and manages the generation process.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load the protogen configuration file."""
    with open(config_path, "r") as f:
        return json.load(f)


def generate_grpc_code(proto_dir: str, output_dir: str, proto_files: list, output_package: str) -> None:
    """Generate gRPC code for the specified proto files."""
    # Create output directories
    output_path = Path(output_dir) / output_package.replace(".", "/")
    output_path.mkdir(parents=True, exist_ok=True)

    # Create __init__.py if it doesn't exist
    init_file = output_path / "__init__.py"
    if not init_file.exists():
        init_file.write_text(f'"""{output_package} gRPC generated modules."""\n\n__all__ = []\n')

    # Run grpc_tools.protoc for each proto file
    proto_path = Path(proto_dir)

    for proto_file in proto_files:
        proto_full_path = proto_path / proto_file
        if not proto_full_path.exists():
            print(f"Warning: Proto file not found: {proto_full_path}")
            continue

        # Generate Python code
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                "-I",
                proto_dir,
                "--python_out",
                output_dir,
                "--grpc_python_out",
                output_dir,
                str(proto_full_path),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Error generating code for {proto_file}:")
            print(result.stderr)
        else:
            print(f"Generated code for {proto_file}")


def main() -> int:
    """Main entry point."""
    config_path = "protogen.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        return 1

    config = load_config(config_path)
    proto_dir = config.get("proto_dir", "./proto")
    output_dir = config.get("output_dir", "./gen/python")

    print(f"Generating gRPC code from {proto_dir} to {output_dir}")

    for package in config.get("packages", []):
        name = package.get("name", "unknown")
        proto_files = package.get("proto_files", [])
        output_package = package.get("output_package", "generated")

        print(f"\nProcessing package: {name}")
        generate_grpc_code(proto_dir, output_dir, proto_files, output_package)

    print("\nGeneration complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())