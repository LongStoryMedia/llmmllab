#!/usr/bin/env python3
"""
Protobuf Generator Script

This script provides a flexible way to generate code from Protocol Buffers (.proto files)
for multiple languages including Python, Go, and others.

Features:
- Supports multiple output languages
- Handles multiple proto files
- Customizable output directories
- Dependency management
- Error handling and verbose output

Usage:
  python protogen.py --languages python,go --proto_files proto/inference.proto
  python protogen.py --languages python --proto_dir proto
  python protogen.py --config protogen.json

Dependencies:
- protoc (Protocol Buffers compiler)
- Language-specific plugins (protoc-gen-go, etc.)
- grpcio-tools (for Python)
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
from typing import Dict, List, Union, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("protogen")


class ProtobufGenerator:
    """Class to manage protobuf code generation for multiple languages."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the protobuf generator.

        Args:
            config: Configuration dictionary with settings
        """
        self.config = config or {}
        self.project_root = os.path.dirname(os.path.abspath(__file__))

        # Default configuration
        self.default_config = {
            "proto_dir": os.path.join(self.project_root, "proto"),
            "proto_files": [],
            "languages": [],
            "output": {
                "python": os.path.join(self.project_root, "inference", "proto"),
                "go": os.path.join(self.project_root, "inference", "proto"),
            },
            "verbose": False
        }

        # Merge default config with provided config
        self._merge_config()

    def _merge_config(self):
        """Merge default config with provided config."""
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
            elif isinstance(value, dict) and isinstance(self.config[key], dict):
                for k, v in value.items():
                    if k not in self.config[key]:
                        self.config[key][k] = v

    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed.

        Returns:
            bool: True if all dependencies are installed, False otherwise.
        """
        logger.info("Checking dependencies...")

        # Check protoc
        if not self._check_command("protoc"):
            logger.error("protoc is not installed. Please install Protocol Buffers compiler.")
            return False

        logger.info(f"✅ protoc found: {shutil.which('protoc')}")

        # Check language-specific dependencies
        if "python" in self.config["languages"]:
            try:
                subprocess.check_call(
                    [sys.executable, "-c", "import grpc_tools.protoc"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                logger.info("✅ grpc_tools.protoc found")
            except (subprocess.CalledProcessError, ImportError):
                logger.error("grpc_tools.protoc not found. Please install with: pip install grpcio-tools")
                return False

        if "go" in self.config["languages"]:
            # Check Go
            if not self._check_command("go"):
                logger.error("Go is not installed. Please install Go.")
                return False

            logger.info(f"✅ Go found: {shutil.which('go')}")

            # Install Go plugins if needed
            self._install_go_dependencies()

        return True

    def _check_command(self, command: str) -> bool:
        """Check if a command exists.

        Args:
            command: Command name to check

        Returns:
            bool: True if command exists, False otherwise
        """
        return shutil.which(command) is not None

    def _install_go_dependencies(self):
        """Install required Go dependencies for protobuf generation."""
        logger.info("Installing required Go packages...")

        packages = [
            "google.golang.org/protobuf/cmd/protoc-gen-go@latest",
            "google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest"
        ]

        for pkg in packages:
            try:
                subprocess.check_call(
                    ["go", "install", pkg],
                    stdout=subprocess.PIPE if not self.config["verbose"] else None,
                    stderr=subprocess.PIPE if not self.config["verbose"] else None
                )
                logger.info(f"✅ Installed {pkg}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to install {pkg}: {e}")

    def _find_proto_files(self) -> List[str]:
        """Find all .proto files in the proto directory.

        Returns:
            List[str]: List of proto file paths
        """
        proto_files = []

        # If specific proto files are provided, use them
        if self.config["proto_files"]:
            for proto_file in self.config["proto_files"]:
                # Check if it's a relative path
                if not os.path.isabs(proto_file):
                    proto_file = os.path.join(self.project_root, proto_file)

                if os.path.exists(proto_file):
                    proto_files.append(proto_file)
                else:
                    logger.warning(f"Proto file not found: {proto_file}")

            if not proto_files:
                logger.error("No valid proto files found in the specified list.")
                return []
        else:
            # Scan the proto directory for .proto files
            proto_dir = self.config["proto_dir"]
            if os.path.exists(proto_dir) and os.path.isdir(proto_dir):
                for file in os.listdir(proto_dir):
                    if file.endswith(".proto"):
                        proto_files.append(os.path.join(proto_dir, file))

            if not proto_files:
                logger.error(f"No .proto files found in directory: {proto_dir}")
                return []

        logger.info(f"Found {len(proto_files)} proto files:")
        for file in proto_files:
            logger.info(f"  - {file}")

        return proto_files

    def generate_python(self, proto_files: List[str]) -> bool:
        """Generate Python code from protobuf files.

        Args:
            proto_files: List of proto file paths

        Returns:
            bool: True if generation was successful, False otherwise
        """
        logger.info("Generating Python code...")

        output_dir = self.config["output"]["python"]
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            sys.executable, "-m", "grpc_tools.protoc",
            f"--proto_path={self.config['proto_dir']}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
        ] + proto_files

        logger.debug(f"Executing command: {' '.join(cmd)}")

        try:
            subprocess.check_call(
                cmd,
                stdout=subprocess.PIPE if not self.config["verbose"] else None,
                stderr=subprocess.PIPE if not self.config["verbose"] else None
            )

            # Create __init__.py
            init_file = os.path.join(output_dir, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write("# Auto-generated package marker\n")

            logger.info(f"Python code generation successful! Output: {output_dir}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate Python code: {e}")
            return False

    def generate_go(self, proto_files: List[str]) -> bool:
        """Generate Go code from protobuf files.

        Args:
            proto_files: List of proto file paths

        Returns:
            bool: True if generation was successful, False otherwise
        """
        logger.info("Generating Go code...")

        output_dir = self.config["output"]["go"]
        os.makedirs(output_dir, exist_ok=True)

        # Get GOPATH
        try:
            go_path = subprocess.check_output(
                ["go", "env", "GOPATH"],
                universal_newlines=True
            ).strip()

            # Add GOPATH/bin to PATH
            os.environ["PATH"] = os.path.join(go_path, "bin") + os.pathsep + os.environ["PATH"]
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get GOPATH: {e}")
            return False

        cmd = [
            "protoc",
            f"--proto_path={self.config['proto_dir']}",
            f"--go_out={output_dir}",
            "--go_opt=paths=source_relative",
            f"--go-grpc_out={output_dir}",
            "--go-grpc_opt=paths=source_relative"
        ] + proto_files

        logger.debug(f"Executing command: {' '.join(cmd)}")

        try:
            subprocess.check_call(
                cmd,
                stdout=subprocess.PIPE if not self.config["verbose"] else None,
                stderr=subprocess.PIPE if not self.config["verbose"] else None
            )

            logger.info(f"Go code generation successful! Output: {output_dir}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate Go code: {e}")
            return False

    def generate(self) -> bool:
        """Generate code for all specified languages.

        Returns:
            bool: True if generation was successful for all languages, False otherwise
        """
        # Check dependencies first
        if not self.check_dependencies():
            logger.error("Missing dependencies. Please install the required tools.")
            return False

        # Find proto files
        proto_files = self._find_proto_files()
        if not proto_files:
            return False

        success = True

        # Generate code for each language
        for language in self.config["languages"]:
            if language == "python":
                if not self.generate_python(proto_files):
                    success = False
            elif language == "go":
                if not self.generate_go(proto_files):
                    success = False
            else:
                logger.warning(f"Unsupported language: {language}")

        return success

    @staticmethod
    def from_arguments(args: argparse.Namespace) -> 'ProtobufGenerator':
        """Create a generator from command-line arguments.

        Args:
            args: Command-line arguments

        Returns:
            ProtobufGenerator: Initialized generator
        """
        config = {}

        # Load config from file if specified
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
        else:
            # Build config from arguments
            config["languages"] = args.languages.split(',') if args.languages else []

            if args.proto_files:
                config["proto_files"] = args.proto_files.split(',')

            if args.proto_dir:
                config["proto_dir"] = args.proto_dir

            if args.python_out:
                if "output" not in config:
                    config["output"] = {}
                config["output"]["python"] = args.python_out

            if args.go_out:
                if "output" not in config:
                    config["output"] = {}
                config["output"]["go"] = args.go_out

            config["verbose"] = args.verbose

        return ProtobufGenerator(config)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Generate code from protobuf files")

    parser.add_argument(
        "--languages",
        type=str,
        help="Comma-separated list of languages to generate (e.g., python,go)"
    )

    parser.add_argument(
        "--proto_files",
        type=str,
        help="Comma-separated list of proto files to process"
    )

    parser.add_argument(
        "--proto_dir",
        type=str,
        help="Directory containing proto files"
    )

    parser.add_argument(
        "--python_out",
        type=str,
        help="Output directory for Python code"
    )

    parser.add_argument(
        "--go_out",
        type=str,
        help="Output directory for Go code"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON config file"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.config and os.path.exists(args.config):
        logger.info(f"Using config file: {args.config}")
    elif not args.languages:
        logger.error("No languages specified. Use --languages or --config.")
        return 1

    generator = ProtobufGenerator.from_arguments(args)

    if generator.generate():
        logger.info("Code generation completed successfully!")
        return 0
    else:
        logger.error("Code generation failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
