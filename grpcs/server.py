"""
gRPC server implementations for ComposerService and RunnerService.

This module provides the server-side implementations for the gRPC services
defined in the grpcs package. It enables:
- Server->Composer communication via ComposerService
- Composer->Runner communication via RunnerService
"""

from typing import Any, AsyncIterator
import grpc as grpcio
from grpc.aio import ServicerContext

from . import (
    server_composer_pb2,
    server_composer_pb2_grpc,
    composer_runner_pb2,
    composer_runner_pb2_grpc,
)


class ComposerServiceServicer(server_composer_pb2_grpc.ComposerServiceServicer):
    """
    gRPC server implementation for ComposerService.

    This service allows the server to:
    - Compose workflows
    - Execute workflows with streaming
    - Create initial workflow state
    - Clear workflow caches
    """

    def __init__(self, composer_service: Any = None):
        """
        Initialize the ComposerServiceServicer.

        Args:
            composer_service: The ComposerService instance to delegate to
        """
        self.composer_service = composer_service

    async def ComposeWorkflow(
        self,
        request: server_composer_pb2.ComposeWorkflowRequest,
        context: ServicerContext,
    ) -> server_composer_pb2.WorkflowHandle:
        """
        Compose a new workflow from requirements.

        Args:
            request: ComposeWorkflowRequest with workflow parameters
            context: gRPC context

        Returns:
            WorkflowHandle with the created workflow ID
        """
        from composer import get_composer_service

        composer = get_composer_service()

        try:
            workflow = await composer.compose_workflow(
                user_id=request.user_id,
                model_name=request.model_name,
                workflow_type=request.workflow_type,
            )
            workflow_id = f"workflow_{id(workflow)}"

            return server_composer_pb2.WorkflowHandle(
                workflow_id=workflow_id,
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    async def ExecuteWorkflow(
        self,
        request: server_composer_pb2.ExecuteWorkflowRequest,
        context: ServicerContext,
    ) -> AsyncIterator[server_composer_pb2.ChatResponse]:
        """
        Execute a workflow with streaming responses.

        Args:
            request: ExecuteWorkflowRequest with workflow ID and initial state
            context: gRPC context

        Yields:
            ChatResponse messages with streaming output
        """
        from composer import get_composer_service
        from runner import pipeline_factory

        composer = get_composer_service()

        try:
            # Compose workflow if not already composed
            workflow = await composer.compose_workflow(
                user_id=request.initial_state.user_id,
                workflow_type=request.initial_state.workflow_type,
            )

            # Simulate streaming output - in production this would stream
            # actual workflow execution events
            response = server_composer_pb2.ChatResponse(
                delta=server_composer_pb2.ChatDelta(
                    message_id="response_1",
                    role="assistant",
                    content="Workflow executed successfully",
                    metadata={},
                )
            )
            yield response

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    async def CreateInitialState(
        self,
        request: server_composer_pb2.CreateInitialStateRequest,
        context: ServicerContext,
    ) -> server_composer_pb2.WorkflowState:
        """
        Create initial state for a workflow.

        Args:
            request: CreateInitialStateRequest with workflow parameters
            context: gRPC context

        Returns:
            WorkflowState with initial state data
        """
        # TODO: Implement actual state creation
        return server_composer_pb2.WorkflowState(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            workflow_type=request.workflow_type,
            variables={},
        )

    async def ClearWorkflowCache(
        self,
        request: server_composer_pb2.ClearWorkflowCacheRequest,
        context: ServicerContext,
    ) -> server_composer_pb2.ClearWorkflowCacheResponse:
        """
        Clear cached workflows for a user.

        Args:
            request: ClearWorkflowCacheRequest with user ID
            context: gRPC context

        Returns:
            ClearWorkflowCacheResponse with results
        """
        # TODO: Implement actual cache clearing
        return server_composer_pb2.ClearWorkflowCacheResponse(
            success=True,
            message="Cache cleared successfully",
            cleared_count=0,
        )


class RunnerServiceServicer(composer_runner_pb2_grpc.RunnerServiceServicer):
    """
    gRPC server implementation for RunnerService.

    This service allows the composer to:
    - Create pipelines from profiles
    - Execute pipelines with streaming output
    - Generate embeddings
    - Get cache statistics
    - Evict pipelines from cache
    """

    def __init__(self, runner_service: Any = None):
        """
        Initialize the RunnerServiceServicer.

        Args:
            runner_service: The RunnerService instance to delegate to
        """
        self.runner_service = runner_service

    async def CreatePipeline(
        self,
        request: composer_runner_pb2.CreatePipelineRequest,
        context: ServicerContext,
    ) -> composer_runner_pb2.PipelineHandle:
        """
        Create a new pipeline from profile.

        Args:
            request: CreatePipelineRequest with model profile and options
            context: gRPC context

        Returns:
            PipelineHandle with the created pipeline ID
        """
        from runner import pipeline_factory

        try:
            # Extract profile information
            profile = request.profile
            model_name = profile.model_name
            provider = profile.provider
            task_type = profile.task_type

            # Create the pipeline
            pipeline = await pipeline_factory.create_pipeline(
                model_name=model_name,
                provider=provider,
                task_type=task_type,
            )

            pipeline_id = f"pipeline_{id(pipeline)}"

            return composer_runner_pb2.PipelineHandle(
                pipeline_id=pipeline_id,
                model_name=model_name,
                is_cached=True,
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    async def ExecutePipeline(
        self,
        request: composer_runner_pb2.ExecutePipelineRequest,
        context: ServicerContext,
    ) -> AsyncIterator[composer_runner_pb2.PipelineEvent]:
        """
        Execute a pipeline with streaming output.

        Args:
            request: ExecutePipelineRequest with pipeline ID and input data
            context: gRPC context

        Yields:
            PipelineEvent messages with streaming output
        """
        # TODO: Implement actual pipeline execution
        yield composer_runner_pb2.PipelineEvent(
            token_chunk=composer_runner_pb2.TokenChunk(
                token="test",
                token_id=123,
                probability=0.95,
                metadata={},
            )
        )

        yield composer_runner_pb2.PipelineEvent(
            complete=composer_runner_pb2.PipelineComplete(
                output_data=b"Pipeline execution complete",
                duration_ms=100,
            )
        )

    async def GenerateEmbeddings(
        self,
        request: composer_runner_pb2.GenerateEmbeddingsRequest,
        context: ServicerContext,
    ) -> composer_runner_pb2.GenerateEmbeddingsResponse:
        """
        Generate embeddings for texts.

        Args:
            request: GenerateEmbeddingsRequest with texts and model info
            context: gRPC context

        Returns:
            GenerateEmbeddingsResponse with embeddings
        """
        # TODO: Implement actual embedding generation
        embeddings = []
        for i, text in enumerate(request.texts):
            # Create dummy embeddings
            embedding_values = [0.1 * (j + 1) for j in range(768)]
            embeddings.append(
                composer_runner_pb2.Embedding(
                    values=embedding_values,
                    index=i,
                )
            )

        return composer_runner_pb2.GenerateEmbeddingsResponse(
            embeddings=embeddings,
            model_dimension=768,
        )

    async def GetCacheStats(
        self,
        request: composer_runner_pb2.GetCacheStatsRequest,
        context: ServicerContext,
    ) -> composer_runner_pb2.CacheStats:
        """
        Get pipeline cache statistics.

        Args:
            request: GetCacheStatsRequest with optional pipeline type filter
            context: gRPC context

        Returns:
            CacheStats with cache information
        """
        from runner import local_pipeline_cache

        try:
            stats = local_pipeline_cache.get_stats()

            return composer_runner_pb2.CacheStats(
                total_pipelines=stats.get("total_pipelines", 0),
                cached_pipelines=stats.get("cached_pipelines", 0),
                active_pipelines=stats.get("active_pipelines", 0),
                total_memory_bytes=stats.get("total_memory_bytes", 0),
                available_memory_bytes=stats.get("available_memory_bytes", 0),
                cache_hits=stats.get("cache_hits", 0),
                cache_misses=stats.get("cache_misses", 0),
                hit_rate=stats.get("hit_rate", 0.0),
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    async def EvictPipeline(
        self,
        request: composer_runner_pb2.EvictPipelineRequest,
        context: ServicerContext,
    ) -> composer_runner_pb2.EvictPipelineResponse:
        """
        Evict a pipeline from cache.

        Args:
            request: EvictPipelineRequest with pipeline ID and force flag
            context: gRPC context

        Returns:
            EvictPipelineResponse with results
        """
        # TODO: Implement actual pipeline eviction
        return composer_runner_pb2.EvictPipelineResponse(
            success=True,
            message="Pipeline evicted successfully",
            freed_memory_bytes=0,
        )


def add_services_to_server(
    composer_servicer: ComposerServiceServicer,
    runner_servicer: RunnerServiceServicer,
    server: Any,
) -> None:
    """Add both ComposerService and RunnerService to a gRPC server"""
    server_composer_pb2_grpc.add_ComposerServiceServicer_to_server(
        composer_servicer, server
    )
    composer_runner_pb2_grpc.add_RunnerServiceServicer_to_server(
        runner_servicer, server
    )


def create_grpc_server(
    composer_servicer: ComposerServiceServicer = None,
    runner_servicer: RunnerServiceServicer = None,
    port: int = 50051,
    runner_port: int = 50052,
) -> Any:
    """
    Create a gRPC server hosting both ComposerService and RunnerService.

    Args:
        composer_servicer: ComposerServiceServicer instance
        runner_servicer: RunnerServiceServicer instance
        port: Port for ComposerService (default: 50051)
        runner_port: Port for RunnerService (default: 50052)

    Returns:
        Configured gRPC Server instance
    """
    server = grpc.aio.server()

    # Create default servicers if not provided
    if composer_servicer is None:
        composer_servicer = ComposerServiceServicer()
    if runner_servicer is None:
        runner_servicer = RunnerServiceServicer()

    # Add services to server
    add_services_to_server(composer_servicer, runner_servicer, server)

    # Add ports
    server.add_insecure_port(f"[::]:{port}")
    server.add_insecure_port(f"[::]:{runner_port}")

    return server


async def start_grpc_server(
    composer_servicer: ComposerServiceServicer = None,
    runner_servicer: RunnerServiceServicer = None,
    port: int = 50051,
    runner_port: int = 50052,
) -> Any:
    """
    Create and start a gRPC server hosting both services.

    Args:
        composer_servicer: ComposerServiceServicer instance
        runner_servicer: RunnerServiceServicer instance
        port: Port for ComposerService (default: 50051)
        runner_port: Port for RunnerService (default: 50052)

    Returns:
        Started gRPC Server instance
    """
    server = create_grpc_server(composer_servicer, runner_servicer, port, runner_port)
    await server.start()
    return server