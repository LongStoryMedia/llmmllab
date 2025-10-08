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
from typing import Dict, Any, Optional, List

from langgraph.graph.state import CompiledStateGraph

from models import (
    Message,
    LangChainMessage,
    MessageRole,
    WorkflowType,
    MessageContent,
    MessageContentType,
)

from composer.graph.state import WorkflowState, ExecutionMetadata
from composer.graph.builder import GraphBuilder
from composer.graph.cache import WorkflowCache
from composer.monitoring.logging import composer_logger
from composer.utils.conversion import (
    convert_messages_to_langchain,
    message_to_langchain_message,
)


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
        self.logger = composer_logger.logger
        from runner import pipeline_factory  # pylint: disable=import-outside-toplevel

        self.pipeline_factory = pipeline_factory

        self.graph_builder = GraphBuilder(pipeline_factory=self.pipeline_factory)
        # Workflow cache is now created per-user during workflow composition
        self.workflow_caches: Dict[str, WorkflowCache] = {}

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
            builder_fn = lambda: self.graph_builder.build_workflow(user_id)

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
        messages: List[Message],
        conversation_id: int = 0,
    ) -> WorkflowState:
        """Create initial workflow state from messages."""

        # Get user configuration from shared data layer
        from db import storage  # pylint: disable=import-outside-toplevel

        user_config = await storage.get_service(storage.user_config).get_user_config(
            user_id
        )
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

        execution_metadata = ExecutionMetadata(
            created_at=asyncio.get_event_loop().time(),
            composer_version="0.1.0",
            streaming_enabled=(
                getattr(user_config.workflow, "enable_streaming", True)
                if hasattr(user_config, "workflow")
                else True
            ),
        )

        # Create the state with centralized user configuration
        state = WorkflowState(
            messages=langchain_messages,
            current_user_message=current_user_message,
            user_id=user_id,
            user_config=user_config,
            conversation_id=conversation_id,
            execution_metadata=execution_metadata,
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
            # Check if streaming is enabled (use user's workflow preference from state metadata)
            streaming_enabled = initial_state.execution_metadata.streaming_enabled
            if stream and streaming_enabled:
                # Stream execution events
                async for event in workflow.astream_events(
                    initial_state.model_dump(), version="v2"
                ):
                    yield event
            else:
                # Batch execution
                result = await workflow.ainvoke(initial_state.model_dump())
                yield {"event": "workflow_complete", "data": result}

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
