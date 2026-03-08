"""
pytest configuration for server tests.

This conftest.py sets up the test environment before pytest collection begins,
ensuring environment variables are available when test modules import server code.
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path

import pytest


def pytest_configure(config):
    """Configure test environment before collection."""
    # Create temp directories for testing
    temp_base = Path(tempfile.gettempdir()) / "llmmll_test"
    temp_base.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    config.temp_dir = temp_base
    config.test_images_dir = temp_base / "images"
    config.test_config_dir = temp_base / "config"
    config.test_cache_dir = temp_base / "cache"

    for dir_path in [config.test_images_dir, config.test_config_dir, config.test_cache_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Set environment variables BEFORE any server imports
    os.environ["HF_TOKEN"] = "test-hf-token"
    os.environ["IMAGE_DIR"] = str(config.test_images_dir)
    os.environ["CONFIG_DIR"] = str(config.test_config_dir)
    os.environ["HF_HOME"] = str(config.test_cache_dir)
    os.environ["TEST_MODE"] = "true"
    os.environ["DISABLE_AUTH"] = "true"
    os.environ["LOG_LEVEL"] = "debug"

    # Clear any cached server imports
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("server"):
            del sys.modules[mod_name]


@pytest.fixture(scope="session")
def test_temp_dir(pytestconfig):
    """Get the session-level temp directory."""
    return pytestconfig.temp_dir


@pytest.fixture(scope="function")
def temp_dir(test_temp_dir, request):
    """Create a function-scoped temp directory, cleaned up after each test."""
    test_dir = test_temp_dir / request.node.name
    test_dir.mkdir(parents=True, exist_ok=True)
    yield test_dir
    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture
def setup_test_environment():
    """Set up test environment variables."""
    # Environment variables are set in pytest_configure
    # This fixture ensures they're available
    pass


@pytest.fixture
def mock_model_profile(mocker):
    """Create a mock model profile."""
    profile = mocker.MagicMock()
    profile.model_name = "test-model"
    profile.temperature = 0.7
    profile.max_tokens = 100
    profile.id = "profile-123"
    return profile


@pytest.fixture
def mock_model(mocker):
    """Create a mock model."""
    model = mocker.MagicMock()
    model.id = "test-model"
    model.name = "test-model"
    model.provider = "LLAMA_CPP"
    model.task = "TEXTTOTEXT"
    model.pipeline = "ChatLlamaCppPipeline"
    return model