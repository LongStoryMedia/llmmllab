"""
Module wrapper for protobuf imports that avoids modifying generated code.
This allows the generated code to find its imports without changes.
"""

import os
import sys
import importlib.util


def find_module(name):
    """
    Find a module by name in various possible locations.
    """
    # Try in the current directory
    if os.path.isfile(f"{name}.py"):
        return f"{name}.py"

    # Try in the proto directory
    proto_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "protos")
    if os.path.isfile(os.path.join(proto_dir, f"{name}.py")):
        return os.path.join(proto_dir, f"{name}.py")

    return None


def setup_proto_imports():
    """
    Set up the import system to find proto modules without modifying generated code.
    """
    # Add proto directory to Python path
    proto_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "protos")
    if proto_dir not in sys.path:
        sys.path.insert(0, proto_dir)

    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    # Ensure parent directory is in path for relative imports
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
