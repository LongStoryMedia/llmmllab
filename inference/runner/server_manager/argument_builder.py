"""
Argument Builder - Structured flag management using argparse for server configurations.

This module provides backwards compatibility imports for the refactored
argument builder components.

DEPRECATED: Use individual modules instead:
- base_argument_builder.BaseArgumentBuilder
- dynamic_flag_parser.DynamicFlagParser  
- llamacpp_argument_builder.LlamaCppArgumentBuilder
- argument_builder_factory.create_argument_builder
"""

# Backwards compatibility imports
from .argument_builder_factory import create_argument_builder
from .base_argument_builder import BaseArgumentBuilder
from .dynamic_flag_parser import DynamicFlagParser
from .llamacpp_argument_builder import LlamaCppArgumentBuilder

__all__ = [
    "BaseArgumentBuilder",
    "DynamicFlagParser",
    "LlamaCppArgumentBuilder",
    "create_argument_builder",
]