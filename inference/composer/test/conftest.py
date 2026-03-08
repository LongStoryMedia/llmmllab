"""
pytest configuration for composer tests.

This conftest.py sets up the test environment for composer component tests.
"""
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

import pytest


def pytest_configure(config):
    """Configure test environment before collection."""
    temp_base = Path(tempfile.gettempdir()) / "llmmll_composer_test"
    temp_base.mkdir(parents=True, exist_ok=True)

    config.temp_dir = temp_base
    config.test_cache_dir = temp_base / "cache"

    for dir_path in [config.test_cache_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(config.test_cache_dir)
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
def mock_graph_builder():
    """Mock GraphBuilder."""
    mock_builder = MagicMock()
    mock_builder.build_workflow = AsyncMock()
    return mock_builder


@pytest.fixture
def mock_server_interface(mocker):
    """Mock ServerInterface."""
    mock_server = MagicMock()

    mock_user_config = MagicMock()
    mock_user_config.get_user_config = AsyncMock()
    mock_server.user_config = mock_user_config

    mock_conversation = MagicMock()
    mock_conversation.get_conversation = AsyncMock()
    mock_server.conversation = mock_conversation

    mock_message = MagicMock()
    mock_message.create_message = AsyncMock()
    mock_server.message = mock_message

    mock_memory = MagicMock()
    mock_memory.search = AsyncMock()
    mock_server.memory = mock_memory

    mock_summary = MagicMock()
    mock_summary.create_summary = AsyncMock()
    mock_server.summary = mock_summary

    mock_model_profile = MagicMock()
    mock_model_profile.get_model_profile = AsyncMock()
    mock_server.model_profile = mock_model_profile

    mock_dynamic_tool = MagicMock()
    mock_dynamic_tool.get_tools = AsyncMock()
    mock_server.dynamic_tool = mock_dynamic_tool

    return mock_server


@pytest.fixture
def composer_service(mock_graph_builder, mock_server_interface):
    """Create a ComposerService instance."""
    from composer.core.service import ComposerService
    return ComposerService(builder=mock_graph_builder, server=mock_server_interface)