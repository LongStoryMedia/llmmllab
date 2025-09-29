"""
Composer Service Interface Layer.

Provides the public API boundary for the composer component, enabling other 
services to interact with composer functionality while maintaining strict 
architectural decoupling. This interface abstracts LangGraph workflow 
construction, execution, and state management.

Interface Functions:
- initialize_composer(): Service lifecycle management
- compose_workflow(): Create executable LangGraph workflows  
- create_initial_state(): Generate workflow state from conversation context
- execute_workflow(): Stream-enabled workflow execution
- get_composer_config(): Runtime configuration access

Architectural Role:
- Defines clean API boundaries between components
- Abstracts internal composer implementation details
- Enables dependency injection for external services
- Maintains Protocol-based decoupling requirements
"""

from typing import Dict, Any, Optional, AsyncGenerator, Union
from composer.core.service import ComposerService
from composer.config import config
from composer.monitoring.logging import composer_logger
from models.conversation_ctx import ConversationCtx

from langchain_core.runnables.schema import StreamEvent

# Global service instance
_composer_service: Optional[ComposerService] = None


async def initialize_composer() -> None:
    """Initialize the composer service. Should be called once at startup."""
    global _composer_service
    if _composer_service is None:
        composer_logger.logger.info("Initializing composer service")
        _composer_service = ComposerService()
        composer_logger.logger.info("Composer service initialized")


async def shutdown_composer() -> None:
    """Shutdown the composer service. Should be called at server shutdown."""
    global _composer_service
    if _composer_service:
        composer_logger.logger.info("Shutting down composer service")
        await _composer_service.shutdown()
        _composer_service = None


def get_composer_service() -> ComposerService:
    """Get the composer service instance."""
    if _composer_service is None:
        raise RuntimeError(
            "Composer service not initialized. Call initialize_composer() first."
        )
    return _composer_service


async def compose_workflow(
    conversation_ctx: ConversationCtx,
    workflow_type: str,
    config_overrides: Optional[Dict[str, Any]] = None,
):
    """
    Compose a workflow for the given conversation context.

    Args:
        conversation_ctx: The conversation context
        workflow_type: Type of workflow ("CHAT", "RESEARCH", etc.)
        config_overrides: Optional configuration overrides

    Returns:
        CompiledStateGraph: Ready to execute LangGraph workflow

    Raises:
        RuntimeError: If composer service not initialized
        WorkflowConstructionError: If workflow construction fails
    """
    service = get_composer_service()
    return await service.compose_workflow(
        conversation_ctx, workflow_type, config_overrides
    )


async def create_initial_state(
    conversation_ctx: ConversationCtx,
    workflow_type: str,
    additional_context: Optional[Dict[str, Any]] = None,
):
    """Create initial workflow state from conversation context."""
    service = get_composer_service()
    return await service.create_initial_state(
        conversation_ctx, workflow_type, additional_context
    )


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
    service = get_composer_service()
    async for event in service.execute_workflow(workflow, initial_state, stream):
        yield event


def get_composer_config():
    """Get current composer configuration."""
    if _composer_service is None:
        raise RuntimeError(
            "Composer service not initialized. Call initialize_composer() first."
        )
    return {
        "service": "composer",
        "caching_enabled": config.default_workflow.enable_workflow_caching,
        "streaming_enabled": config.default_workflow.enable_streaming,
        "multi_agent_enabled": config.default_workflow.enable_multi_agent,
        "tool_generation_enabled": config.default_tool.enable_tool_generation,
        "cache_ttl": config.default_workflow.workflow_cache_ttl,
        "max_context_length": config.default_workflow.max_context_length,
        "tool_similarity_threshold": config.default_tool.tool_similarity_threshold,
    }


# Convenience exports for direct usage
__all__ = [
    "initialize_composer",
    "shutdown_composer",
    "get_composer_service",
    "compose_workflow",
    "create_initial_state",
    "execute_workflow",
    "get_composer_config",
]
