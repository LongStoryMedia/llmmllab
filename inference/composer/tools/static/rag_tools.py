"""
DEPRECATED: LangChain tool wrappers with architectural violations.

This file violates architectural decoupling by importing from server components.
Use native_rag_tools.py instead, which implements proper Protocol-based 
dependency injection without cross-component imports.

The classes in this file will be removed in a future version.
"""

# This file is deprecated - use native_rag_tools.py instead
# DO NOT ADD NEW FUNCTIONALITY HERE

import warnings

warnings.warn(
    "rag_tools.py is deprecated due to architectural violations. "
    "Use native_rag_tools.py with dependency injection instead.",
    DeprecationWarning,
    stacklevel=2
)
