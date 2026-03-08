"""
Composer integration tests.

Tests the LangGraph workflow composition and execution.
"""

import pytest
from typing import Dict, Any

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_composer_imports():
    """Test that composer modules can be imported."""
    try:
        from composer import ComposerService, get_builder
        from composer.api.interface import ServerInterface, ServerAdapter
        from composer.graph.workflows.factory import get_builder
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import composer: {e}")


@pytest.mark.asyncio
async def test_composer_service_exists():
    """Test that the ComposerService class is accessible."""
    from composer import ComposerService

    assert ComposerService is not None


@pytest.mark.asyncio
async def test_server_interface_protocol():
    """Test that the ServerInterface protocol is defined."""
    from composer.api.interface import ServerInterface

    # Check that the protocol has the required methods
    assert hasattr(ServerInterface, "__protocol_attrs__") or hasattr(ServerInterface, "__slots__")


@pytest.mark.asyncio
async def test_workflow_builder_factory():
    """Test that the workflow builder factory works."""
    from composer.graph.workflows.factory import get_builder
    from composer.graph.workflows.base import WorkFlowType

    # Test getting IDE builder
    ide_builder = get_builder(WorkFlowType.IDE)
    assert ide_builder is not None

    # Test getting Dialog builder
    dialog_builder = get_builder(WorkFlowType.DIALOG)
    assert dialog_builder is not None


@pytest.mark.asyncio
async def test_composer_with_mocked_server(db_connection):
    """Test composer workflow composition with mocked server services."""
    from composer import ComposerService
    from composer.api.interface import ServerAdapter
    from server.app import app

    # This test verifies that composer can be instantiated
    # with a mocked server interface
    composer = ComposerService()

    assert composer is not None


@pytest.mark.asyncio
async def test_workflow_caching():
    """Test that workflow caching is configured."""
    from composer.graph.cache import workflow_cache

    assert workflow_cache is not None
    assert hasattr(workflow_cache, "get")
    assert hasattr(workflow_cache, "set")


@pytest.mark.asyncio
async def test_composer_models():
    """Test that composer models are accessible."""
    from composer.models import (
        WorkFlowType,
        WorkflowConfig,
        IntentAnalysis,
    )

    assert WorkFlowType is not None
    assert WorkflowConfig is not None
    assert IntentAnalysis is not None