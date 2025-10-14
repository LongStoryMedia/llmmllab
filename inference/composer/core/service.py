"""
Main ComposerService orchestrator.
Central to the redesign - serves as the primary, authoritative execution runtime.

Configuration Management:
- Configuration overrides and default merging happens at the data layer
- Configuration is NOT passed as arguments in composer components
- Allowed arguments: user_id, messages/query, tools, workflow_type
- Components retrieve configuration from shared data layer using user_id
- No configuration merging logic should exist in service layer components
"""

import asyncio
from typing import Dict, Optional, TYPE_CHECKING
from datetime import datetime, timezone

from langgraph.graph.state import CompiledStateGraph

from models import (
    Message,
    MessageRole,
    UserConfig,
    MessageContent,
    MessageContentType,
)

from composer.graph.state import WorkflowState
from composer.graph.builder import GraphBuilder
from composer.graph.cache import WorkflowCache
from utils.logging import llmmllogger
from composer.utils.conversion import (
    convert_messages_to_langchain,
    message_to_langchain_message,
)


if TYPE_CHECKING:
    from composer.graph.builder import GraphBuilder


class ComposerService:
    """
    Main composer service coordinating graph construction and execution.

    The Composer is responsible for:
    - Graph construction & execution
    - Streaming orchestration
    - State management
    - Tool management
    - Intent analysis
    - Error resiliency
    - Multi-agent orchestration
    """

    def __init__(self):
        self.logger = llmmllogger.logger
        from runner import pipeline_factory  # pylint: disable=import-outside-toplevel

        self.pipeline_factory = pipeline_factory
        self.storage = None
        self.graph_builder: Optional["GraphBuilder"] = None
        # Workflow cache is now created per-user during workflow composition
        self.workflow_caches: Dict[str, WorkflowCache] = {}

    def _ensure_graph_builder(self, user_config: UserConfig) -> None:
        """Lazily create GraphBuilder when needed, ensuring storage is available."""
        if self.graph_builder is None:
            from db import storage  # pylint: disable=import-outside-toplevel

            if not storage.initialized:
                raise RuntimeError(
                    "Storage must be initialized before using ComposerService"
                )

            self.storage = storage
            self.graph_builder = GraphBuilder(
                storage,
                self.pipeline_factory,
                user_config,
            )

        # Assert for type checking that graph_builder is not None after this call
        assert self.graph_builder is not None

    async def compose_workflow(
        self,
        user_id: str,
    ) -> CompiledStateGraph:
        """
        Construct or retrieve a master workflow with intelligent routing.

        The workflow will handle intent analysis, tool selection, and routing
        internally using LangGraph's native capabilities.

        args:
            user_id: User ID for configuration retrieval

        returns:
            CompiledStateGraph: Master workflow with intelligent routing
        """
        try:
            # 1. Get user configuration from shared data layer
            from db import storage  # pylint: disable=import-outside-toplevel

            user_config = await storage.get_service(
                storage.user_config
            ).get_user_config(user_id)

            # 2. Use per-user cache if enabled (cache based on user_id only now)
            user_cache = None
            if user_config.workflow.enable_workflow_caching:
                if user_id not in self.workflow_caches:
                    self.workflow_caches[user_id] = WorkflowCache()
                user_cache = self.workflow_caches[user_id]

                # Simplified cache key - master workflow is the same for all users
                cache_key = f"master_workflow_{user_id}"

                cached_workflow = await user_cache.get(cache_key)
                if cached_workflow:
                    self.logger.debug(
                        "Retrieved master workflow from cache",
                        extra={"cache_key": cache_key},
                    )
                    return cached_workflow

            # 3. Build master workflow with intelligent routing or explicit type
            # Intent analysis and tool selection happen inside the graph now
            self._ensure_graph_builder(user_config)

            # Type guard: assert graph_builder is available after _ensure_graph_builder
            graph_builder = self.graph_builder
            assert graph_builder is not None, "GraphBuilder should be initialized"

            builder_fn = lambda: graph_builder.build_workflow(user_id)

            if user_cache:
                workflow = await user_cache.get_or_create(cache_key, builder_fn)
            else:
                workflow = await builder_fn()

            self.logger.info(
                "Master workflow composed successfully", extra={"user_id": user_id}
            )

            return workflow

        except Exception as e:
            self.logger.error(
                "Failed to compose master workflow",
                extra={"error": str(e), "user_id": user_id},
                exc_info=True,
            )
            raise

    async def create_initial_state(
        self,
        user_id: str,
        conversation_id: int,
    ) -> WorkflowState:
        """Create initial workflow state from messages."""

        # Get user configuration from shared data layer
        from db import storage  # pylint: disable=import-outside-toplevel

        user_config = await storage.get_service(storage.user_config).get_user_config(
            user_id
        )

        messages = await storage.get_service(storage.message).get_conversation_history(
            conversation_id
        )

        summaries = await storage.get_service(
            storage.summary
        ).get_summaries_for_conversation(conversation_id)

        langchain_messages = convert_messages_to_langchain(messages)

        current_user_message = message_to_langchain_message(
            next(
                (msg for msg in reversed(messages) if msg.role == MessageRole.USER),
                Message(
                    content=[
                        MessageContent(type=MessageContentType.TEXT, text="", url=None)
                    ],
                    role=MessageRole.USER,
                ),
            )
        )

        # Create the state with centralized user configuration
        state = WorkflowState(
            messages=langchain_messages,
            summaries=summaries,
            current_user_message=current_user_message,
            user_id=user_id,
            user_config=user_config,
            conversation_id=conversation_id,
        )

        return state

    async def execute_workflow(
        self,
        workflow: CompiledStateGraph,
        initial_state: WorkflowState,
        stream: bool = True,
    ):
        """
        Execute a compiled workflow with the given initial state.

        Supports both streaming and batch execution modes.
        """
        try:
            async for event in workflow.astream_events(
                initial_state.model_dump(), version="v2"
            ):
                try:
                    # Inject tool_calls and node metadata into event data if present in state but missing in event
                    if isinstance(event, dict):
                        data = event.get("data")
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
                                    event["data"] = new_data  # type: ignore[index]

                            # Also check if the event itself has node information we can enrich
                            event_name = event.get("name", "")
                            event_type = event.get("event", "")

                            # Add execution metadata to certain event types for better traceability
                            if event_type in [
                                "on_chain_start",
                                "on_chain_end",
                                "on_tool_start",
                                "on_tool_end",
                            ]:
                                if "metadata" not in event:
                                    event["metadata"] = {}  # type: ignore[index]

                                # Add timing and context information
                                event["metadata"].update(
                                    {  # type: ignore[index]
                                        "timestamp": datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                        "workflow_context": "composer_service",
                                    }
                                )

                            # Else if top-level tool_calls already emitted by node update, keep as-is
                    yield event
                except Exception as e:
                    self.logger.warning(
                        "Error enriching workflow event",
                        extra={
                            "error": str(e),
                            "event_type": event.get("event", "unknown"),
                        },
                    )
                    # On any injection error, still yield original event to avoid stream disruption
                    yield event

        except Exception as e:
            self.logger.error(
                "Workflow execution failed", extra={"error": str(e)}, exc_info=True
            )
            yield {"event": "workflow_error", "data": {"error": str(e)}}

    async def shutdown(self):
        """Clean up resources on service shutdown."""
        self.logger.info("Shutting down ComposerService")

        # Close all per-user workflow caches
        for user_id, cache in self.workflow_caches.items():
            try:
                await cache.close()
            except Exception as e:
                self.logger.warning(f"Error closing cache for user {user_id}: {e}")
        self.workflow_caches.clear()
        # Close other resources as needed
