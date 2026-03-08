"""
Runner integration tests.

Tests the pipeline factory and model execution.
"""

import pytest
from typing import Dict, Any

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_runner_imports():
    """Test that runner modules can be imported."""
    try:
        from runner import pipeline_factory, local_pipeline_cache
        from runner.pipelines.llamacpp import ChatLlamaCppPipeline
        from runner.utils import hardware_manager
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import runner: {e}")


@pytest.mark.asyncio
async def test_pipeline_factory_exists():
    """Test that the pipeline factory is accessible."""
    from runner import pipeline_factory

    assert pipeline_factory is not None
    assert hasattr(pipeline_factory, "create_pipeline")


@pytest.mark.asyncio
async def test_local_pipeline_cache_exists():
    """Test that the local pipeline cache is accessible."""
    from runner import local_pipeline_cache

    assert local_pipeline_cache is not None
    assert hasattr(local_pipeline_cache, "get")
    assert hasattr(local_pipeline_cache, "set")


@pytest.mark.asyncio
async def test_hardware_manager_exists():
    """Test that the hardware manager is accessible."""
    from runner.utils import hardware_manager

    assert hardware_manager is not None
    assert hasattr(hardware_manager, "get_gpu_info")


@pytest.mark.asyncio
async def test_pipeline_types():
    """Test that pipeline types are defined."""
    from runner.pipelines.base import Pipeline
    from runner.pipelines.llamacpp import ChatLlamaCppPipeline
    from runner.pipelines.embed import EmbedLlamaCppPipeline

    assert Pipeline is not None
    assert ChatLlamaCppPipeline is not None
    assert EmbedLlamaCppPipeline is not None


@pytest.mark.asyncio
async def test_runner_models():
    """Test that runner models are accessible."""
    from runner.models import (
        ModelProfile,
        ModelProvider,
        ModelTask,
        PipelineConfig,
    )

    assert ModelProfile is not None
    assert ModelProvider is not None
    assert ModelTask is not None
    assert PipelineConfig is not None


@pytest.mark.asyncio
async def test_pipeline_cache_stats():
    """Test that pipeline cache statistics can be retrieved."""
    from runner import local_pipeline_cache

    stats = local_pipeline_cache.get_cache_stats()

    assert stats is not None
    assert "cache_size" in stats
    assert "max_cache_size" in stats


@pytest.mark.asyncio
async def test_gpu_detection():
    """Test that GPU detection works (if GPU is available)."""
    from runner.utils import hardware_manager

    gpu_info = hardware_manager.get_gpu_info()

    # Should return a dict or list of GPUs
    assert gpu_info is not None
    if isinstance(gpu_info, list):
        # If GPUs are detected, check structure
        for gpu in gpu_info:
            assert "name" in gpu or "id" in gpu