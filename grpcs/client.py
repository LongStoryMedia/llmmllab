# Client implementations for gRPC services
# Uses absolute imports for grpcio to avoid circular issues
# Uses relative imports for proto modules within grpcs package

import grpc as grpcio
from grpc.aio import Channel, UnaryUnaryCall, UnaryStreamCall

from . import server_composer_pb2, server_composer_pb2_grpc
from . import composer_runner_pb2, composer_runner_pb2_grpc


class ServerComposerClient:
    """Client for the ComposerService (Server -> Composer communication)"""

    def __init__(self, channel: Channel):
        self._channel = channel
        self._stub = server_composer_pb2_grpc.ComposerServiceStub(channel)

    async def compose_workflow(
        self,
        user_id: str,
        workflow_type: str,
        model_name: str,
        timestamp,
        user_config: dict[str, str] | None = None
    ) -> server_composer_pb2.WorkflowHandle:
        """Create a new workflow from requirements"""
        request = server_composer_pb2.ComposeWorkflowRequest(
            user_id=user_id,
            workflow_type=workflow_type,
            model_name=model_name,
            timestamp=timestamp,
            user_config=user_config or {}
        )
        return await self._stub.ComposeWorkflow(request)

    async def execute_workflow(
        self,
        workflow_id: str,
        initial_state: server_composer_pb2.WorkflowState,
        stream_events: bool = True
    ) -> UnaryStreamCall:
        """Execute a workflow with streaming responses"""
        request = server_composer_pb2.ExecuteWorkflowRequest(
            workflow_id=workflow_id,
            initial_state=initial_state,
            stream_events=stream_events
        )
        return self._stub.ExecuteWorkflow(request)

    async def create_initial_state(
        self,
        user_id: str,
        conversation_id: int,
        workflow_type: str
    ) -> server_composer_pb2.WorkflowState:
        """Create initial state for a workflow"""
        request = server_composer_pb2.CreateInitialStateRequest(
            user_id=user_id,
            conversation_id=conversation_id,
            workflow_type=workflow_type
        )
        return await self._stub.CreateInitialState(request)

    async def clear_workflow_cache(
        self,
        user_id: str
    ) -> server_composer_pb2.ClearWorkflowCacheResponse:
        """Clear cached workflows for a user"""
        request = server_composer_pb2.ClearWorkflowCacheRequest(user_id=user_id)
        return await self._stub.ClearWorkflowCache(request)


class ComposerRunnerClient:
    """Client for the RunnerService (Composer -> Runner communication)"""

    def __init__(self, channel: Channel):
        self._channel = channel
        self._stub = composer_runner_pb2_grpc.RunnerServiceStub(channel)

    async def create_pipeline(
        self,
        profile: composer_runner_pb2.ModelProfile,
        priority: str = "normal",
        grammar_type: str = "auto",
        metadata: dict[str, str] | None = None
    ) -> composer_runner_pb2.PipelineHandle:
        """Create a new pipeline from profile"""
        request = composer_runner_pb2.CreatePipelineRequest(
            profile=profile,
            priority=priority,
            grammar_type=grammar_type,
            metadata=metadata or {}
        )
        return await self._stub.CreatePipeline(request)

    async def execute_pipeline(
        self,
        pipeline_id: str,
        input_data: bytes,
        stream_output: bool = True
    ) -> UnaryStreamCall:
        """Execute a pipeline with streaming output"""
        request = composer_runner_pb2.ExecutePipelineRequest(
            pipeline_id=pipeline_id,
            input_data=input_data,
            stream_output=stream_output
        )
        return self._stub.ExecutePipeline(request)

    async def generate_embeddings(
        self,
        texts: list[str],
        model_name: str,
        dimension: int | None = None
    ) -> composer_runner_pb2.GenerateEmbeddingsResponse:
        """Generate embeddings for texts"""
        request = composer_runner_pb2.GenerateEmbeddingsRequest(
            texts=texts,
            model_name=model_name,
            dimension=dimension or 0
        )
        return await self._stub.GenerateEmbeddings(request)

    async def get_cache_stats(
        self,
        pipeline_type: str = ""
    ) -> composer_runner_pb2.CacheStats:
        """Get pipeline cache statistics"""
        request = composer_runner_pb2.GetCacheStatsRequest(pipeline_type=pipeline_type)
        return await self._stub.GetCacheStats(request)

    async def evict_pipeline(
        self,
        pipeline_id: str,
        force: bool = False
    ) -> composer_runner_pb2.EvictPipelineResponse:
        """Evict a pipeline from cache"""
        request = composer_runner_pb2.EvictPipelineRequest(
            pipeline_id=pipeline_id,
            force=force
        )
        return await self._stub.EvictPipeline(request)