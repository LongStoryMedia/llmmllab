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
from langchain_core.runnables.config import RunnableConfig

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
from composer.graph.executor import WorkflowExecutor
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
        # Generic workflow executor for streaming
        self.executor = WorkflowExecutor(
            logger=self.logger, default_context="composer_service"
        )

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

        # Load active todos for continuation context
        active_todos = await storage.get_service(
            storage.todo
        ).get_todos_by_conversation(user_id, conversation_id)

        # Create the state with centralized user configuration and todo context
        state = WorkflowState(
            messages=langchain_messages,
            summaries=summaries,
            current_user_message=current_user_message,
            user_id=user_id,
            user_config=user_config,
            conversation_id=conversation_id,
            active_todos=active_todos,  # Include active todos for context continuity
            checkpoint_metadata={
                "conversation_id": conversation_id,
                "user_id": user_id,
                "turn_timestamp": datetime.now(timezone.utc).isoformat(),
            },
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
        Now uses the generic WorkflowExecutor for consistent behavior.
        """
        # Create thread ID for checkpointing
        thread_id = f"thread_{initial_state.user_id}_{initial_state.conversation_id}"

        if stream:
            # Use generic streaming executor
            async for event in self.executor.stream_workflow(
                workflow=workflow,
                initial_state=initial_state,
                thread_id=thread_id,
                context_name="composer_service",
            ):
                yield event
        else:
            # Use batch execution mode
            try:
                result = await self.executor.run_workflow(
                    workflow=workflow,
                    initial_state=initial_state,
                    thread_id=thread_id,
                )
                yield {"event": "workflow_complete", "data": result}
            except Exception as e:
                self.logger.error(
                    "Batch workflow execution failed",
                    extra={"error": str(e)},
                    exc_info=True,
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
