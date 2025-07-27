#!/usr/bin/env python3
"""
Wrapper script to run the gRPC server with proper import paths set up.
This avoids the need to modify generated protobuf code.
"""

import grpc_server
import os
import sys
import argparse

# Set up the Python path properly - crucial to get imports right
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add both directories to path to ensure proper import resolution
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# Print current Python path for debugging
print("Python path:")
for path in sys.path:
    print(f"  {path}")

# Print available proto modules for debugging
print("\nChecking proto module availability:")
proto_dir = os.path.join(current_dir, "protos")
if os.path.exists(proto_dir):
    print(f"Proto directory exists at: {proto_dir}")
    for file in os.listdir(proto_dir):
        if file.endswith(".py") and not file.startswith("__"):
            print(f"  Found proto module: {file}")

# Create __init__.py files if they don't exist to ensure proper package structure
os.makedirs(os.path.join(current_dir, "protos"), exist_ok=True)
init_file = os.path.join(current_dir, "protos", "__init__.py")
if not os.path.exists(init_file):
    with open(init_file, 'w') as f:
        f.write('"""Proto package."""\n')

print("\nStarting gRPC server with proper import paths...")

# Import and run the gRPC server

# Run the server
if __name__ == "__main__":
    # Parse command line arguments like the original script
    parser = argparse.ArgumentParser(description="Inference gRPC Server")
    parser.add_argument(
        "--port", type=int, default=50051, help="Port to listen on (default: 50051)"
    )
    args = parser.parse_args()

    # Create required directories
    from config import IMAGE_DIR, CONFIG_DIR
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)

    # Call the server with the parsed port
    grpc_server.serve(args.port)
