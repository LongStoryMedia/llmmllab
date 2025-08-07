#!/usr/bin/env python3
"""
Script to regenerate protobuf files with proper configuration.
This helps ensure compatibility between Python and gRPC versions.
"""

import os
import sys
import subprocess


def main():
    print("Regenerating protobuf files...")

    # Define paths
    proto_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "protos")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "protos")

    # Create directories if they don't exist
    os.makedirs(proto_dir, exist_ok=True)

    # Find all .proto files
    proto_files = [f for f in os.listdir(proto_dir) if f.endswith('.proto')]

    if not proto_files:
        print("No .proto files found in the proto directory.")
        return 1

    # Generate Python code for each proto file
    for proto_file in proto_files:
        proto_path = os.path.join(proto_dir, proto_file)
        cmd = [
            "python", "-m", "grpc_tools.protoc",
            f"--proto_path={proto_dir}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            proto_path
        ]

        print(f"Generating code for {proto_file}...")
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            print(f"Successfully generated code for {proto_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error generating code for {proto_file}:")
            print(f"Command: {' '.join(cmd)}")
            print(f"Error: {e.stderr.decode('utf-8')}")
            return 1

    # Fix imports in generated files to ensure compatibility
    print("Fixing imports in generated files...")
    for file in os.listdir(output_dir):
        if file.endswith('_pb2.py') or file.endswith('_pb2_grpc.py'):
            file_path = os.path.join(output_dir, file)
            with open(file_path, 'r') as f:
                content = f.read()

            # Fix relative imports
            if '_pb2_grpc.py' in file:
                # Fix imports in _pb2_grpc files
                original = f'import {file.replace("_grpc.py", "")}'
                replacement = f'from . import {file.replace("_grpc.py", "")}'
                content = content.replace(original, replacement)

            with open(file_path, 'w') as f:
                f.write(content)

    # Update __init__.py
    init_path = os.path.join(output_dir, "__init__.py")
    with open(init_path, "w") as f:
        f.write('"""\nProtocol buffer module initialization.\n"""\n\n')
        f.write("# Import all protobuf modules\n\n")

        for proto_file in proto_files:
            module_name = os.path.splitext(proto_file)[0]
            f.write(f"from . import {module_name}_pb2\n")
            f.write(f"from . import {module_name}_pb2_grpc\n")

        f.write("\n# Export all modules\n")
        f.write("__all__ = [\n")
        for proto_file in proto_files:
            module_name = os.path.splitext(proto_file)[0]
            f.write(f"    '{module_name}_pb2', '{module_name}_pb2_grpc',\n")
        f.write("]\n")

    print("Proto regeneration complete!")
    print("Please restart your application to use the new proto files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
