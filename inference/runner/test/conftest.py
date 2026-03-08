"""
pytest configuration for runner tests.

This conftest.py sets up the test environment for runner component tests.
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
    temp_base = Path(tempfile.gettempdir()) / "llmmll_runner_test"
    temp_base.mkdir(parents=True, exist_ok=True)

    config.temp_dir = temp_base
    config.test_cache_dir = temp_base / "cache"
    config.test_models_dir = temp_base / "models"

    for dir_path in [config.test_cache_dir, config.test_models_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Set environment variables
    os.environ["HF_HOME"] = str(config.test_cache_dir)
    os.environ["MODELS_FILE_PATH"] = str(config.test_models_dir / "models.yaml")
    os.environ["TEST_MODE"] = "true"


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
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture
def mock_pipeline(mocker):
    """Create a mock pipeline."""
    pipeline = mocker.MagicMock()
    pipeline.execute = mocker.AsyncMock()
    pipeline.generate = mocker.AsyncMock()
    return pipeline


@pytest.fixture
def mock_model_profile(mocker):
    """Create a mock model profile."""
    profile = mocker.MagicMock()
    profile.model_name = "test-model"
    profile.temperature = 0.7
    profile.max_tokens = 100
    profile.id = "profile-123"
    profile.provider = "LLAMA_CPP"
    profile.task = "TEXTTOTEXT"
    return profile