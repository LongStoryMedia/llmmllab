"""
Temporary file to print available SQL query keys to help with debugging.
"""

import logging
import sys
import os

# Add the parent directory to the path so we can import from server
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from server.db.queries import get_loader

# Set up logging
logging.basicConfig(level=logging.INFO)

# Get the loader and print all available keys
loader = get_loader()
print("Available SQL query keys:")
for key in sorted(loader.queries.keys()):
    print(f"- {key}")
