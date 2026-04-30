"""
Composer Service Interface Layer.

Provides the public API boundary for the composer component, enabling other
services to interact with composer functionality while maintaining strict
architectural decoupling. This interface abstracts LangGraph workflow
construction, execution, and state management.

Interface Functions:
- initialize_composer(): Service lifecycle management
- compose_workflow(): Create executable LangGraph workflows using user_id and messages
- create_initial_state(): Generate workflow state from user_id and messages
- execute_workflow(): Stream-enabled workflow execution
- get_composer_config(): Runtime configuration access

Architectural Role:
- Defines clean API boundaries between components
- Abstracts internal composer implementation details
- Enables dependency injection for external services
- Maintains Protocol-based decoupling requirements
"""

from typing import AsyncIterator, List, Optional, Type, Union
import uuid
from pydantic import BaseModel
from models import ChatResponse, Message
from utils.logging import llmmllogger
from core.service import CompiledStateGraph, ComposerService
from core.errors import ComposerError
from graph.executor import stream_workflow
from graph.state import ServerToolEvent
from graph.workflows.base import GraphBuilder
from graph.workflows.factory import WorkFlowType, get_builder


class ComposerServiceManager:
    """Singleton manager for composer service instance."""

    _instance: Optional["ComposerServiceManager"] = None
    _service: Optional[ComposerService] = None

    def __new__(cls) -> "ComposerServiceManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def initialize(self, builder: GraphBuilder) -> None:
        """Initialize the composer service. Should be called once at startup."""
        if self._service is None:
            llmmllogger.logger.info("Initializing composer service")
            self._service = ComposerService(builder)
            llmmllogger.logger.info("Composer service initialized")

    async def shutdown(self) -> None:
        """Shutdown the composer service. Should be called at server shutdown."""
        if self._service:
            llmmllogger.logger.info("Shutting down composer service")
            await self._service.shutdown()
            self._service = None

    def get_service(self) -> ComposerService:
        """Get the composer service instance."""
        if self._service is None:
            raise RuntimeError(
                "Composer service not initialized. Call initialize_composer() first."
            )
        return self._service

    async def get_or_init_service(
        self, builder: Optional[GraphBuilder] = None
    ) -> ComposerService:
        """Get or initialize the composer service instance."""
        if self._service is None:
            if builder is None:
                raise ComposerError(
                    "WorkflowBuilder is required for composer service initialization."
                )
            await self.initialize(builder)
        assert self._service is not None
        return self._service


_manager = ComposerServiceManager()


async def shutdown_composer() -> None:
    """Shutdown the composer service. Should be called at server shutdown."""
    await _manager.shutdown()


async def get_or_init_composer_service(builder: GraphBuilder) -> ComposerService:
    """Get or initialize the composer service instance."""
    return await _manager.get_or_init_service(builder)


async def compose_workflow(
    user_id: str,
    builder: GraphBuilder,
    model_name: Optional[str] = None,
    response_format: Optional[Type[BaseModel]] = None,
    **build_kwargs,
) -> CompiledStateGraph:
    """
    Compose a workflow for the given user and conversation messages.

    Args:
        user_id: User ID for configuration retrieval from shared data layer
        builder: GraphBuilder instance
        response_format: Optional response format constraint
        **build_kwargs: Additional keyword arguments passed to build_workflow
            (e.g., client_tools, tool_choice for IDE workflows)

    Returns:
        CompiledStateGraph: Ready to execute LangGraph workflow
    """
    svc = await _manager.get_or_init_service(builder)
    return await svc.compose_workflow(
        user_id=user_id,
        model_name=model_name,
        response_format=response_format,
        **build_kwargs,
    )


async def clear_workflow_cache(user_id: str) -> None:
    """
    Clear the workflow cache for a specific user.

    Args:
        user_id: User ID whose workflow cache should be cleared
    """
    try:
        svc = await _manager.get_or_init_service()
        cache = svc.workflow_caches.get(user_id, None)
        if cache:
            await cache.close()
    except ComposerError as e:
        llmmllogger.logger.error(
            f"Error clearing workflow cache for user {user_id}: {e}"
        )


async def create_initial_state(
    user_id: str,
    conversation_id: int,
    builder: GraphBuilder,
    messages: Optional[List[Message]] = None,
):
    """Create initial workflow state from user messages and configuration.

    Args:
        user_id: User ID for configuration retrieval from shared data layer
        messages: List of conversation messages
        workflow_type: Type of workflow
        additional_context: Optional additional context for state initialization

    Returns:
        WorkflowState: Initial state for workflow execution

    Note:
        User configuration is retrieved from shared data layer using user_id.
        No configuration objects should be passed as arguments (architectural rule).
    """
    return await builder.create_initial_state(user_id, conversation_id, messages)


async def execute_workflow(
    initial_state: BaseModel,
    workflow: CompiledStateGraph,
) -> AsyncIterator[Union[ChatResponse, ServerToolEvent]]:
    """
    Execute a compiled workflow with the given initial state.

    Args:
        workflow: CompiledStateGraph from compose_workflow()
        initial_state: WorkflowState from create_initial_state()
        stream: Whether to stream events or return final result

    Yields:
        Dict containing workflow events (tokens, state updates, etc.)
    """
    async for event in stream_workflow(initial_state, workflow, thread_id=str(uuid.uuid4())):
        yield event


async def get_graph_builder(workflow_type: WorkFlowType, user_id: str) -> GraphBuilder:
    """Get the workflow builder instance. Should be implemented by external service."""
    return await get_builder(workflow_type, user_id)


# Convenience exports for direct usage
__all__ = [
    "shutdown_composer",
    "compose_workflow",
    "create_initial_state",
    "execute_workflow",
]
