"""
Configuration for the composer service.

This module provides configuration values needed by composer components,
with fallback to server.config for shared values like IMAGE_DIR.
"""

import os
from typing import Any, Dict


def get_image_dir() -> str:
    """Get the image directory, falling back to server.config if needed."""
    return os.environ.get("IMAGE_DIR", "/root/images")


def get_image_retention_hours() -> int:
    """Get the image retention period in hours."""
    return int(os.environ.get("IMAGE_RETENTION_HOURS", "24"))


# Expose commonly needed config values
IMAGE_DIR = get_image_dir()
IMAGE_RETENTION_HOURS = get_image_retention_hours()