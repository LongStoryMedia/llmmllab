"""
pytest configuration for unit tests.

This conftest.py sets up the test environment before pytest collection begins,
ensuring environment variables are available when test modules import server code.
"""
import os

# Set environment variables at module level (before any pytest fixtures run)
# This ensures they are available when test modules import server code during collection

# Create temp directories for testing
os.makedirs("/tmp/test_images", exist_ok=True)
os.makedirs("/tmp/test_config", exist_ok=True)

# Set environment variables BEFORE any server imports
os.environ["HF_TOKEN"] = "test-token"
os.environ["IMAGE_DIR"] = "/tmp/test_images"
os.environ["CONFIG_DIR"] = "/tmp/test_config"
os.environ["HF_HOME"] = "/tmp/test_hf_cache"