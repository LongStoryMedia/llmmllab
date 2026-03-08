"""
Unit tests for composer/core/service.py.

Tests ComposerService orchestration of graph construction and execution.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock, call
from typing import Optional, Type
import uuid

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from composer.core.service import ComposerService
from composer.models import (
    UserConfig,
    ModelProfileConfig,
    WorkflowConfig,
    CircuitBreakerConfig,
    GPUConfig,
    ParameterOptimizationConfig,
    CrashPrevention,
)


@pytest.fixture
def mock_graph_builder(mocker):
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

    return mock_server


@pytest.fixture
def composer_service(mock_graph_builder, mock_server_interface):
    """Create a ComposerService instance."""
    return ComposerService(builder=mock_graph_builder, server=mock_server_interface)


class TestComposerServiceInitialization:
    """Tests for ComposerService initialization."""

    def test_initialization_with_builder_and_server(self, mock_graph_builder, mock_server_interface):
        """Test service initialization with builder and server."""
        service = ComposerService(builder=mock_graph_builder, server=mock_server_interface)

        assert service.graph_builder == mock_graph_builder
        assert service.server == mock_server_interface
        assert service.workflow_caches == {}

    def test_initialization_without_server(self, mock_graph_builder):
        """Test service initialization without server."""
        service = ComposerService(builder=mock_graph_builder)

        assert service.graph_builder == mock_graph_builder
        assert service.server is None
        assert service.workflow_caches == {}


class TestComposeWorkflow:
    """Tests for compose_workflow method."""

    @pytest.mark.asyncio
    async def test_compose_workflow_success(self, composer_service, mock_graph_builder, mock_server_interface):
        """Test successful workflow composition."""
        mock_user_config = UserConfig(
            user_id="user-1",
            model_profiles=ModelProfileConfig(
                primary_profile_id=uuid.uuid4(),
                summarization_profile_id=uuid.uuid4(),
                master_summary_profile_id=uuid.uuid4(),
                brief_summary_profile_id=uuid.uuid4(),
                key_points_profile_id=uuid.uuid4(),
                improvement_profile_id=uuid.uuid4(),
                analysis_profile_id=uuid.uuid4(),
                memory_retrieval_profile_id=uuid.uuid4(),
                self_critique_profile_id=uuid.uuid4(),
                research_task_profile_id=uuid.uuid4(),
                research_plan_profile_id=uuid.uuid4(),
                research_consolidation_profile_id=uuid.uuid4(),
                research_analysis_profile_id=uuid.uuid4(),
                embedding_profile_id=uuid.uuid4(),
                formatting_profile_id=uuid.uuid4(),
                image_generation_prompt_profile_id=uuid.uuid4(),
                image_generation_profile_id=uuid.uuid4(),
                engineering_profile_id=uuid.uuid4(),
                reranking_profile_id=uuid.uuid4(),
            ),
            workflow=WorkflowConfig(enable_workflow_caching=False),
            circuit_breaker=CircuitBreakerConfig(),
            gpu_config=GPUConfig(),
            parameter_optimization=ParameterOptimizationConfig(
                enabled=False,
                parameters=[],
                crash_prevention=CrashPrevention(),
            ),
        )
        mock_server_interface.user_config.get_user_config.return_value = mock_user_config
        mock_graph_builder.build_workflow.return_value = MagicMock(spec=CompiledStateGraph)

        workflow = await composer_service.compose_workflow("user-1")

        assert workflow is not None
        mock_graph_builder.build_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_compose_workflow_without_server_raises(self, composer_service, mock_graph_builder):
        """Test compose_workflow raises error when server is None."""
        service = ComposerService(builder=mock_graph_builder)
        service.server = None

        with pytest.raises(RuntimeError, match="Server interface is required"):
            await service.compose_workflow("user-1")

    @pytest.mark.asyncio
    async def test_compose_workflow_with_caching(self, mocker, mock_graph_builder, mock_server_interface):
        """Test workflow composition with caching enabled."""
        mock_user_config = UserConfig(
            user_id="user-1",
            model_profiles=ModelProfileConfig(
                primary_profile_id=uuid.uuid4(),
                summarization_profile_id=uuid.uuid4(),
                master_summary_profile_id=uuid.uuid4(),
                brief_summary_profile_id=uuid.uuid4(),
                key_points_profile_id=uuid.uuid4(),
                improvement_profile_id=uuid.uuid4(),
                analysis_profile_id=uuid.uuid4(),
                memory_retrieval_profile_id=uuid.uuid4(),
                self_critique_profile_id=uuid.uuid4(),
                research_task_profile_id=uuid.uuid4(),
                research_plan_profile_id=uuid.uuid4(),
                research_consolidation_profile_id=uuid.uuid4(),
                research_analysis_profile_id=uuid.uuid4(),
                embedding_profile_id=uuid.uuid4(),
                formatting_profile_id=uuid.uuid4(),
                image_generation_prompt_profile_id=uuid.uuid4(),
                image_generation_profile_id=uuid.uuid4(),
                engineering_profile_id=uuid.uuid4(),
                reranking_profile_id=uuid.uuid4(),
            ),
            workflow=WorkflowConfig(enable_workflow_caching=True),
            circuit_breaker=CircuitBreakerConfig(),
            gpu_config=GPUConfig(),
            parameter_optimization=ParameterOptimizationConfig(
                enabled=False,
                parameters=[],
                crash_prevention=CrashPrevention(),
            ),
        )
        mock_server_interface.user_config.get_user_config.return_value = mock_user_config

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.get_or_create = AsyncMock(return_value=MagicMock(spec=CompiledStateGraph))

        mocker.patch('composer.core.service.WorkflowCache', return_value=mock_cache)

        service = ComposerService(builder=mock_graph_builder, server=mock_server_interface)
        mock_graph_builder.build_workflow.return_value = MagicMock(spec=CompiledStateGraph)

        workflow = await service.compose_workflow("user-1")

        assert workflow is not None
        mock_cache.get_or_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_compose_workflow_retrieves_from_cache(self, mocker, mock_graph_builder, mock_server_interface):
        """Test workflow composition retrieves cached workflow."""
        mock_user_config = UserConfig(
            user_id="user-1",
            model_profiles=ModelProfileConfig(
                primary_profile_id=uuid.uuid4(),
                summarization_profile_id=uuid.uuid4(),
                master_summary_profile_id=uuid.uuid4(),
                brief_summary_profile_id=uuid.uuid4(),
                key_points_profile_id=uuid.uuid4(),
                improvement_profile_id=uuid.uuid4(),
                analysis_profile_id=uuid.uuid4(),
                memory_retrieval_profile_id=uuid.uuid4(),
                self_critique_profile_id=uuid.uuid4(),
                research_task_profile_id=uuid.uuid4(),
                research_plan_profile_id=uuid.uuid4(),
                research_consolidation_profile_id=uuid.uuid4(),
                research_analysis_profile_id=uuid.uuid4(),
                embedding_profile_id=uuid.uuid4(),
                formatting_profile_id=uuid.uuid4(),
                image_generation_prompt_profile_id=uuid.uuid4(),
                image_generation_profile_id=uuid.uuid4(),
                engineering_profile_id=uuid.uuid4(),
                reranking_profile_id=uuid.uuid4(),
            ),
            workflow=WorkflowConfig(enable_workflow_caching=True),
            circuit_breaker=CircuitBreakerConfig(),
            gpu_config=GPUConfig(),
            parameter_optimization=ParameterOptimizationConfig(
                enabled=False,
                parameters=[],
                crash_prevention=CrashPrevention(),
            ),
        )
        mock_server_interface.user_config.get_user_config.return_value = mock_user_config

        mock_cached_workflow = MagicMock(spec=CompiledStateGraph)

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=mock_cached_workflow)
        mocker.patch('composer.core.service.WorkflowCache', return_value=mock_cache)

        service = ComposerService(builder=mock_graph_builder, server=mock_server_interface)

        workflow = await service.compose_workflow("user-1")

        assert workflow == mock_cached_workflow
        mock_cache.get.assert_called_once()
        mock_graph_builder.build_workflow.assert_not_called()

    @pytest.mark.asyncio
    async def test_compose_workflow_with_model_name(self, mocker, mock_graph_builder, mock_server_interface):
        """Test workflow composition with model name parameter."""
        mock_user_config = UserConfig(
            user_id="user-1",
            model_profiles=ModelProfileConfig(
                primary_profile_id=uuid.uuid4(),
                summarization_profile_id=uuid.uuid4(),
                master_summary_profile_id=uuid.uuid4(),
                brief_summary_profile_id=uuid.uuid4(),
                key_points_profile_id=uuid.uuid4(),
                improvement_profile_id=uuid.uuid4(),
                analysis_profile_id=uuid.uuid4(),
                memory_retrieval_profile_id=uuid.uuid4(),
                self_critique_profile_id=uuid.uuid4(),
                research_task_profile_id=uuid.uuid4(),
                research_plan_profile_id=uuid.uuid4(),
                research_consolidation_profile_id=uuid.uuid4(),
                research_analysis_profile_id=uuid.uuid4(),
                embedding_profile_id=uuid.uuid4(),
                formatting_profile_id=uuid.uuid4(),
                image_generation_prompt_profile_id=uuid.uuid4(),
                image_generation_profile_id=uuid.uuid4(),
                engineering_profile_id=uuid.uuid4(),
                reranking_profile_id=uuid.uuid4(),
            ),
            workflow=WorkflowConfig(enable_workflow_caching=True),
            circuit_breaker=CircuitBreakerConfig(),
            gpu_config=GPUConfig(),
            parameter_optimization=ParameterOptimizationConfig(
                enabled=False,
                parameters=[],
                crash_prevention=CrashPrevention(),
            ),
        )
        mock_server_interface.user_config.get_user_config.return_value = mock_user_config

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.get_or_create = AsyncMock(return_value=MagicMock(spec=CompiledStateGraph))
        mocker.patch('composer.core.service.WorkflowCache', return_value=mock_cache)

        service = ComposerService(builder=mock_graph_builder, server=mock_server_interface)
        mock_graph_builder.build_workflow.return_value = MagicMock(spec=CompiledStateGraph)

        workflow = await service.compose_workflow("user-1", model_name="gpt-4")

        assert workflow is not None

    @pytest.mark.asyncio
    async def test_compose_workflow_with_response_format(self, composer_service, mock_graph_builder, mock_server_interface):
        """Test workflow composition with response format."""
        class TestResponse(BaseModel):
            field: str

        mock_user_config = UserConfig(
            user_id="user-1",
            model_profiles=ModelProfileConfig(
                primary_profile_id=uuid.uuid4(),
                summarization_profile_id=uuid.uuid4(),
                master_summary_profile_id=uuid.uuid4(),
                brief_summary_profile_id=uuid.uuid4(),
                key_points_profile_id=uuid.uuid4(),
                improvement_profile_id=uuid.uuid4(),
                analysis_profile_id=uuid.uuid4(),
                memory_retrieval_profile_id=uuid.uuid4(),
                self_critique_profile_id=uuid.uuid4(),
                research_task_profile_id=uuid.uuid4(),
                research_plan_profile_id=uuid.uuid4(),
                research_consolidation_profile_id=uuid.uuid4(),
                research_analysis_profile_id=uuid.uuid4(),
                embedding_profile_id=uuid.uuid4(),
                formatting_profile_id=uuid.uuid4(),
                image_generation_prompt_profile_id=uuid.uuid4(),
                image_generation_profile_id=uuid.uuid4(),
                engineering_profile_id=uuid.uuid4(),
                reranking_profile_id=uuid.uuid4(),
            ),
            workflow=WorkflowConfig(enable_workflow_caching=False),
            circuit_breaker=CircuitBreakerConfig(),
            gpu_config=GPUConfig(),
            parameter_optimization=ParameterOptimizationConfig(
                enabled=False,
                parameters=[],
                crash_prevention=CrashPrevention(),
            ),
        )
        mock_server_interface.user_config.get_user_config.return_value = mock_user_config
        mock_graph_builder.build_workflow.return_value = MagicMock(spec=CompiledStateGraph)

        workflow = await composer_service.compose_workflow("user-1", response_format=TestResponse)

        assert workflow is not None
        mock_graph_builder.build_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_compose_workflow_with_build_kwargs(self, composer_service, mock_graph_builder, mock_server_interface):
        """Test workflow composition with build kwargs."""
        mock_user_config = UserConfig(
            user_id="user-1",
            model_profiles=ModelProfileConfig(
                primary_profile_id=uuid.uuid4(),
                summarization_profile_id=uuid.uuid4(),
                master_summary_profile_id=uuid.uuid4(),
                brief_summary_profile_id=uuid.uuid4(),
                key_points_profile_id=uuid.uuid4(),
                improvement_profile_id=uuid.uuid4(),
                analysis_profile_id=uuid.uuid4(),
                memory_retrieval_profile_id=uuid.uuid4(),
                self_critique_profile_id=uuid.uuid4(),
                research_task_profile_id=uuid.uuid4(),
                research_plan_profile_id=uuid.uuid4(),
                research_consolidation_profile_id=uuid.uuid4(),
                research_analysis_profile_id=uuid.uuid4(),
                embedding_profile_id=uuid.uuid4(),
                formatting_profile_id=uuid.uuid4(),
                image_generation_prompt_profile_id=uuid.uuid4(),
                image_generation_profile_id=uuid.uuid4(),
                engineering_profile_id=uuid.uuid4(),
                reranking_profile_id=uuid.uuid4(),
            ),
            workflow=WorkflowConfig(enable_workflow_caching=False),
            circuit_breaker=CircuitBreakerConfig(),
            gpu_config=GPUConfig(),
            parameter_optimization=ParameterOptimizationConfig(
                enabled=False,
                parameters=[],
                crash_prevention=CrashPrevention(),
            ),
        )
        mock_server_interface.user_config.get_user_config.return_value = mock_user_config
        mock_graph_builder.build_workflow.return_value = MagicMock(spec=CompiledStateGraph)

        workflow = await composer_service.compose_workflow("user-1", tools=[], intent="chat")

        assert workflow is not None
        call_kwargs = mock_graph_builder.build_workflow.call_args.kwargs
        assert call_kwargs["tools"] == []
        assert call_kwargs["intent"] == "chat"


class TestShutdown:
    """Tests for shutdown method."""

    @pytest.mark.asyncio
    async def test_shutdown_closes_caches(self, mocker, mock_graph_builder, mock_server_interface):
        """Test shutdown closes all user caches."""
        mock_cache1 = MagicMock()
        mock_cache1.close = AsyncMock()
        mock_cache2 = MagicMock()
        mock_cache2.close = AsyncMock()

        service = ComposerService(builder=mock_graph_builder, server=mock_server_interface)
        service.workflow_caches = {
            "user-1": mock_cache1,
            "user-2": mock_cache2,
        }

        await service.shutdown()

        assert mock_cache1.close.await_count == 1
        assert mock_cache2.close.await_count == 1
        assert service.workflow_caches == {}

    @pytest.mark.asyncio
    async def test_shutdown_handles_cache_error(self, mocker, mock_graph_builder, mock_server_interface):
        """Test shutdown handles cache close errors."""
        mock_cache = MagicMock()
        mock_cache.close = AsyncMock(side_effect=Exception("Close failed"))

        service = ComposerService(builder=mock_graph_builder, server=mock_server_interface)
        service.workflow_caches = {"user-1": mock_cache}

        # Should not raise
        await service.shutdown()

        assert service.workflow_caches == {}