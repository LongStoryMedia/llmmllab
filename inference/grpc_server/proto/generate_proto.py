"""
Script to generate Python code from protobuf definitions.
"""

import os
import sys
import subprocess

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define paths - need to handle paths correctly
# Get the absolute path to this script
SCRIPT_PATH = os.path.abspath(__file__)
# Get the directory of this script
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
# Go up to grpc_server directory
GRPC_SERVER_DIR = os.path.dirname(SCRIPT_DIR)
# Go up to inference directory
INFERENCE_DIR = os.path.dirname(GRPC_SERVER_DIR)
# Go up to project root
PROJECT_ROOT = os.path.dirname(INFERENCE_DIR)
# Proto directory is directly under project root
PROTO_DIR = os.path.join(PROJECT_ROOT, "proto")
PROTO_FILE = os.path.join(PROTO_DIR, "inference.proto")
OUTPUT_DIR = SCRIPT_DIR  # Output to current directory (proto dir)

# Print debug info
print(f"Script path: {SCRIPT_PATH}")
print(f"Script directory: {SCRIPT_DIR}")
print(f"GRPC Server directory: {GRPC_SERVER_DIR}")
print(f"Inference directory: {INFERENCE_DIR}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Proto directory: {PROTO_DIR}")
print(f"Proto file: {PROTO_FILE}")
print(f"Output directory: {OUTPUT_DIR}")


def generate_proto():
    """Generate Python code from protobuf definitions."""
    print(f"Generating Python code from {PROTO_FILE} to {OUTPUT_DIR}")

    # Check if proto file exists
    if not os.path.exists(PROTO_FILE):
        print(f"Error: Proto file not found at {PROTO_FILE}")
        print(f"Current directory: {os.getcwd()}")
        print(f"PROTO_DIR: {PROTO_DIR}")
        print(f"PROJECT_ROOT: {PROJECT_ROOT}")
        sys.exit(1)

    print(f"Proto file found: {PROTO_FILE}")

    # Run the protoc command
    cmd = [
        "python", "-m", "grpc_tools.protoc",
        f"--proto_path={PROTO_DIR}",
        f"--python_out={OUTPUT_DIR}",
        f"--grpc_python_out={OUTPUT_DIR}",
        PROTO_FILE
    ]

    print(f"Executing command: {' '.join(cmd)}")

    # Execute the command
    try:
        subprocess.check_call(cmd)
        print("Proto code generation successful!")

        # Create an empty __init__.py file to mark this directory as a Python package
        init_file = os.path.join(OUTPUT_DIR, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write("# Auto-generated package marker\n")
            print(f"Created package marker: {init_file}")

        print("Python package setup complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error generating proto code: {e}")
        sys.exit(1)


if __name__ == "__main__":
    generate_proto()
