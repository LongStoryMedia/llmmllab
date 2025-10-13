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

from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from langchain_core.runnables.schema import StreamEvent

from models import Message, WorkflowType

from utils.logging import llmmllogger

from .core.service import CompiledStateGraph, ComposerService

# Global service instance
_composer_service: Optional[ComposerService] = None


async def initialize_composer() -> None:
    """Initialize the composer service. Should be called once at startup."""
    global _composer_service  # noqa: PLW0603
    if _composer_service is None:
        llmmllogger.logger.info("Initializing composer service")
        _composer_service = ComposerService()
        llmmllogger.logger.info("Composer service initialized")


async def shutdown_composer() -> None:
    """Shutdown the composer service. Should be called at server shutdown."""
    global _composer_service  # noqa: PLW0603
    if _composer_service:
        llmmllogger.logger.info("Shutting down composer service")
        await _composer_service.shutdown()
        _composer_service = None


def get_composer_service() -> ComposerService:
    """Get the composer service instance."""
    if _composer_service is None:
        raise RuntimeError(
            "Composer service not initialized. Call initialize_composer() first."
        )
    return _composer_service


async def get_or_init_composer_service() -> ComposerService:
    """Get or initialize the composer service instance."""
    global _composer_service  # noqa: PLW0603
    if _composer_service is None:
        await initialize_composer()
    assert _composer_service is not None
    return _composer_service


async def compose_workflow(user_id: str) -> "CompiledStateGraph":
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
    svc = await get_or_init_composer_service()
    return await svc.compose_workflow(user_id)


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
    service = get_composer_service()
    return await service.create_initial_state(user_id, conversation_id)


async def execute_workflow(
    workflow, initial_state, stream: bool = True
) -> AsyncGenerator[Union[StreamEvent, Dict[str, Any]], None]:
    """
    Execute a compiled workflow with the given initial state.

    Args:
        workflow: CompiledStateGraph from compose_workflow()
        initial_state: WorkflowState from create_initial_state()
        stream: Whether to stream events or return final result

    Yields:
        Dict containing workflow events (tokens, state updates, etc.)
    """
    service = await get_or_init_composer_service()
    async for event in service.execute_workflow(workflow, initial_state, stream):
        yield event


# Convenience exports for direct usage
__all__ = [
    "initialize_composer",
    "shutdown_composer",
    "get_composer_service",
    "compose_workflow",
    "create_initial_state",
    "execute_workflow",
]
