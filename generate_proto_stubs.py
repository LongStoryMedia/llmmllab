#!/usr/bin/env python3
"""
Helper script to generate IDE type hints for protobuf generated classes.
This script creates stub files (.pyi) that help IDEs recognize the generated protobuf types.
"""

import os
import sys
import importlib
import inspect
from typing import List, Dict, Any, Optional


def create_stub_for_module(module_name: str, output_dir: str):
    """Create a .pyi stub file for a protobuf module.

    Args:
        module_name: The name of the module to create a stub for
        output_dir: The directory where the stub file will be created
    """
    try:
        module = importlib.import_module(module_name)

        # Get the file path of the module
        module_file = module.__file__
        if not module_file:
            print(f"Could not determine file path for module {module_name}")
            return

        # Determine the stub file name
        module_basename = os.path.basename(module_file)
        stub_file = os.path.splitext(module_basename)[0] + ".pyi"
        stub_path = os.path.join(output_dir, stub_file)

        # Get all message classes defined in the module
        message_classes = []
        for name, obj in inspect.getmembers(module):
            # Protobuf message classes typically have a DESCRIPTOR attribute
            if hasattr(obj, 'DESCRIPTOR') and not name.startswith('_'):
                message_classes.append(name)

        # Create stub file content
        stub_content = [
            f"# Generated stub file for {module_name}",
            "from google.protobuf import descriptor as _descriptor",
            "from google.protobuf import message as _message",
            "from typing import ClassVar, Iterable, Mapping, Optional, Text, Union, List, Dict, Any",
            ""
        ]

        # Add class stubs for each message type
        for cls_name in message_classes:
            stub_content.append(f"class {cls_name}(_message.Message):")
            stub_content.append(f"    DESCRIPTOR: ClassVar[_descriptor.Descriptor]  # type: ignore[override]")
            stub_content.append(f"    def __init__(self, **kwargs) -> None: ...")
            stub_content.append("")

        # Write the stub file
        with open(stub_path, 'w') as f:
            f.write('\n'.join(stub_content))

        print(f"Created stub file: {stub_path}")

    except ImportError as e:
        print(f"Error importing module {module_name}: {e}")
    except Exception as e:
        print(f"Error creating stub for {module_name}: {e}")


def main():
    """Create stub files for all proto modules."""
    # Define the proto modules to create stubs for
    proto_modules = [
        'inference.protos.chat_req_pb2',
        'inference.protos.chat_message_pb2',
        'inference.protos.embedding_req_pb2',
        'inference.protos.embedding_response_pb2',
        'inference.protos.inference_pb2_grpc',
        # Add other modules as needed
    ]

    # Output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference/proto")

    print(f"Creating stub files in {output_dir}")
    for module_name in proto_modules:
        create_stub_for_module(module_name, output_dir)

    print("\nDone! Now IDEs should be able to recognize the proto types.")
    print("Note: You may need to restart your IDE for the changes to take effect.")


if __name__ == "__main__":
    main()
