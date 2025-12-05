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

from typing import AsyncIterator, Optional
from pydantic import BaseModel
from models import ChatResponse
from utils.logging import llmmllogger
from .core.service import CompiledStateGraph, ComposerService
from .graph.executor import stream_workflow


class ComposerServiceManager:
    """Singleton manager for composer service instance."""

    _instance: Optional["ComposerServiceManager"] = None
    _service: Optional[ComposerService] = None

    def __new__(cls) -> "ComposerServiceManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def initialize(self) -> None:
        """Initialize the composer service. Should be called once at startup."""
        if self._service is None:
            llmmllogger.logger.info("Initializing composer service")
            self._service = ComposerService()
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

    async def get_or_init_service(self) -> ComposerService:
        """Get or initialize the composer service instance."""
        if self._service is None:
            await self.initialize()
        assert self._service is not None
        return self._service


_manager = ComposerServiceManager()


async def shutdown_composer() -> None:
    """Shutdown the composer service. Should be called at server shutdown."""
    await _manager.shutdown()


async def get_or_init_composer_service() -> ComposerService:
    """Get or initialize the composer service instance."""
    return await _manager.get_or_init_service()


async def compose_workflow(user_id: str) -> CompiledStateGraph:
    """
    Compose a workflow for the given user and conversation messages.

    Args:
        user_id: User ID for configuration retrieval from shared data layer

    Returns:
        CompiledStateGraph: Ready to execute LangGraph workflow

    Raises:
        RuntimeError: If composer service not initialized
        WorkflowConstructionError: If workflow construction fails

    Note:
        Configuration is retrieved from shared data layer using user_id.
        No configuration objects should be passed as arguments (architectural rule).
    """
    svc = await _manager.get_or_init_service()
    return await svc.compose_workflow(user_id)


async def clear_workflow_cache(user_id: str) -> None:
    """
    Clear the workflow cache for a specific user.

    Args:
        user_id: User ID whose workflow cache should be cleared
    """
    svc = await _manager.get_or_init_service()
    cache = svc.workflow_caches.get(user_id, None)
    if cache:
        await cache.close()


async def create_initial_state(
    user_id: str,
    conversation_id: int,
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
    service = await _manager.get_or_init_service()
    return await service.create_initial_state(user_id, conversation_id)


async def execute_workflow(
    initial_state: BaseModel,
    workflow: CompiledStateGraph,
) -> AsyncIterator[ChatResponse]:
    """
    Execute a compiled workflow with the given initial state.

    Args:
        workflow: CompiledStateGraph from compose_workflow()
        initial_state: WorkflowState from create_initial_state()
        stream: Whether to stream events or return final result

    Yields:
        Dict containing workflow events (tokens, state updates, etc.)
    """
    async for event in stream_workflow(initial_state, workflow):
        yield event


# Convenience exports for direct usage
__all__ = [
    "shutdown_composer",
    "compose_workflow",
    "create_initial_state",
    "execute_workflow",
]
