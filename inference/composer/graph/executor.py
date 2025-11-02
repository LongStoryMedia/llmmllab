"""
Generic workflow execution module for streaming CompiledStateGraph outputs.

This module provides reusable workflow execution capabilities that can be used
across different graph types and state models, extracting the streaming logic
from ComposerService into a generic, reusable component.
"""

from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generic,
    Optional,
    TypeVar,
    Union,
)
from datetime import datetime, timezone

from pydantic import BaseModel

from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.schema import StreamEvent, EventData

from utils.logging import llmmllogger

StateT = TypeVar("StateT", bound=Union[Dict[str, Any], BaseModel])


class WorkflowExecutor(Generic[StateT]):
    """
    Generic workflow executor for CompiledStateGraph streaming.

    Provides reusable streaming execution capabilities that can handle
    any CompiledStateGraph with any state type, as long as the state
    can be converted to a dictionary format.
    """

    def __init__(
        self, logger: Optional[Any] = None, default_context: str = "workflow_executor"
    ):
        """
        Initialize the workflow executor.

        Args:
            logger: Optional logger instance. If None, uses default llmmllogger
            default_context: Default context string for metadata enrichment
        """
        self.logger = logger or llmmllogger.logger
        self.default_context = default_context

    def create_thread_config(
        self, thread_id: str, additional_config: Optional[Dict[str, Any]] = None
    ) -> RunnableConfig:
        """
        Create a thread configuration for workflow checkpointing.

        Args:
            thread_id: Unique thread identifier for checkpointing
            additional_config: Additional configuration parameters

        Returns:
            RunnableConfig: Configuration for LangGraph execution
        """
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        if additional_config:
            config.setdefault("configurable", {}).update(additional_config)

        return config

    async def stream_workflow(
        self,
        workflow: CompiledStateGraph,
        initial_state: StateT,
        config: Optional[RunnableConfig] = None,
        thread_id: Optional[str] = None,
        enrich_events: bool = True,
        context_name: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any] | StreamEvent, None]:
        """
        Execute a compiled workflow with streaming output.

        This method provides generic streaming execution for any CompiledStateGraph
        and state type, with optional event enrichment and error handling.

        Args:
            workflow: CompiledStateGraph to execute
            initial_state: Initial state for workflow execution (dict or convertible object)
            config: Optional RunnableConfig. If None and thread_id provided, creates default config
            thread_id: Thread ID for checkpointing (used if config is None)
            enrich_events: Whether to enrich events with metadata and tool_calls
            context_name: Context name for metadata (defaults to default_context)

        Yields:
            Dict[str, Any]: Stream events from workflow execution
        """
        try:
            # Prepare state for execution
            if isinstance(initial_state, dict):
                state_dict = initial_state
            else:
                # Assume state has model_dump method (Pydantic-like)
                if hasattr(initial_state, "model_dump"):
                    state_dict = initial_state.model_dump()
                else:
                    raise ValueError(
                        f"State type {type(initial_state)} must be dict or have model_dump/dict method"
                    )

            # Create config if not provided
            if config is None and thread_id is not None:
                config = self.create_thread_config(thread_id)

            # Stream workflow events
            async for event in workflow.astream_events(
                state_dict, config=config, version="v2"
            ):
                try:
                    if enrich_events:
                        event = self._enrich_event(
                            event, context_name or self.default_context
                        )

                    yield event

                except Exception as e:
                    self.logger.warning(
                        "Error enriching workflow event",
                        extra={
                            "error": str(e),
                            "event_type": event.get("event", "unknown"),
                        },
                    )
                    # On any enrichment error, still yield original event to avoid stream disruption
                    yield event

        except Exception as e:
            self.logger.error(
                "Workflow execution failed", extra={"error": str(e)}, exc_info=True
            )
            yield {"event": "workflow_error", "data": {"error": str(e)}}

    def _enrich_event(self, event: StreamEvent, context_name: str) -> StreamEvent:
        """
        Enrich workflow events with additional metadata and tool information.

        Args:
            event: Original event from workflow execution
            context_name: Context name for metadata

        Returns:
            Dict[str, Any]: Enriched event
        """
        if not isinstance(event, dict):
            return event

        data: EventData = event.get("data")

        # Events that carry a full state snapshot expose 'values'; prefer that
        if data and isinstance(data, dict):
            # If state serialization present
            state_values = data.get("values") or data.get("state")
            if state_values and isinstance(state_values, dict):
                # Create a shallow copy to avoid mutating a typed dict structure
                new_data = dict(data)
                updated = False

                # Inject tool_calls if missing
                tc = state_values.get("tool_calls")
                if tc and "tool_calls" not in data:
                    new_data["tool_calls"] = tc
                    updated = True

                # Inject node metadata if available
                node_metadata = state_values.get("node_metadata")
                if node_metadata and "node_metadata" not in data:
                    new_data["node_metadata"] = node_metadata
                    updated = True

                # Apply enriched data if we made changes
                if updated:
                    event["data"] = new_data  # type: ignore

        # Also check if the event itself has node information we can enrich
        event_type = event.get("event", "")

        # Add execution metadata to certain event types for better traceability
        if event_type in [
            "on_chain_start",
            "on_chain_end",
            "on_tool_start",
            "on_tool_end",
        ]:
            if "metadata" not in event:
                event["metadata"] = {}

            # Add timing and context information
            event["metadata"].update(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "workflow_context": context_name,
                }
            )

        return event

    async def run_workflow(
        self,
        workflow: CompiledStateGraph,
        initial_state: Union[StateT, Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a compiled workflow in batch mode (non-streaming).

        Args:
            workflow: CompiledStateGraph to execute
            initial_state: Initial state for workflow execution
            config: Optional RunnableConfig
            thread_id: Thread ID for checkpointing

        Returns:
            Dict[str, Any]: Final workflow result
        """
        try:
            # Prepare state for execution
            if isinstance(initial_state, dict):
                state_dict = initial_state
            else:
                if hasattr(initial_state, "model_dump"):
                    state_dict = initial_state.model_dump()
                elif hasattr(initial_state, "dict"):
                    state_dict = initial_state.dict()
                else:
                    raise ValueError(
                        f"State type {type(initial_state)} must be dict or have model_dump/dict method"
                    )

            # Create config if not provided
            if config is None and thread_id is not None:
                config = self.create_thread_config(thread_id)

            # Execute workflow and return final result
            result = await workflow.ainvoke(state_dict, config=config)
            return result

        except Exception as e:
            self.logger.error(
                "Batch workflow execution failed",
                extra={"error": str(e)},
                exc_info=True,
            )
            raise


# Convenience factory functions
def create_executor(
    logger: Optional[Any] = None, context: str = "workflow_executor"
) -> WorkflowExecutor:
    """
    Create a new WorkflowExecutor instance.

    Args:
        logger: Optional logger instance
        context: Default context name

    Returns:
        WorkflowExecutor: New executor instance
    """
    return WorkflowExecutor(logger=logger, default_context=context)


async def stream_workflow(
    workflow: CompiledStateGraph,
    initial_state: Union[StateT, Dict[str, Any]],
    thread_id: Optional[str] = None,
    config: Optional[RunnableConfig] = None,
    logger: Optional[Any] = None,
    context: str = "workflow_stream",
) -> AsyncGenerator[StreamEvent | Dict[str, Any], None]:
    """
    Convenience function for streaming workflow execution.

    Args:
        workflow: CompiledStateGraph to execute
        initial_state: Initial state for workflow execution
        thread_id: Thread ID for checkpointing
        config: Optional RunnableConfig
        logger: Optional logger instance
        context: Context name for metadata

    Yields:
        Dict[str, Any]: Stream events from workflow execution
    """
    executor = create_executor(logger=logger, context=context)

    async for event in executor.stream_workflow(
        workflow=workflow,
        initial_state=initial_state,
        config=config,
        thread_id=thread_id,
        context_name=context,
    ):
        yield event


async def run_workflow(
    workflow: CompiledStateGraph,
    initial_state: Union[StateT, Dict[str, Any]],
    thread_id: Optional[str] = None,
    config: Optional[RunnableConfig] = None,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Convenience function for batch workflow execution.

    Args:
        workflow: CompiledStateGraph to execute
        initial_state: Initial state for workflow execution
        thread_id: Thread ID for checkpointing
        config: Optional RunnableConfig
        logger: Optional logger instance

    Returns:
        Dict[str, Any]: Final workflow result
    """
    executor = create_executor(logger=logger)

    return await executor.run_workflow(
        workflow=workflow,
        initial_state=initial_state,
        config=config,
        thread_id=thread_id,
    )
