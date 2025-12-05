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

from typing import Dict, Optional
from datetime import datetime, timezone

from langgraph.graph.state import CompiledStateGraph

from models import (
    Message,
    MessageRole,
    MessageContent,
    MessageContentType,
)

from composer.graph.state import WorkflowState
from composer.graph.builder import GraphBuilder
from composer.graph.cache import WorkflowCache
from composer.graph.executor import WorkflowExecutor
from utils.logging import llmmllogger


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
        self.graph_builder: Optional["GraphBuilder"] = None
        # Workflow cache is now created per-user during workflow composition
        self.workflow_caches: Dict[str, WorkflowCache] = {}
        # Generic workflow executor for streaming
        self.executor = WorkflowExecutor(
            logger=self.logger, default_context="composer_service"
        )

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
                cache_key = f"workflow_{user_id}"

                cached_workflow = await user_cache.get(cache_key)
                if cached_workflow:
                    self.logger.debug(
                        "Retrieved workflow from cache",
                        extra={"cache_key": cache_key},
                    )
                    return cached_workflow

            # 3. Build master workflow
            graph_builder = self.graph_builder = GraphBuilder(
                storage,
                self.pipeline_factory,
                user_config,
            )
            assert graph_builder is not None, "GraphBuilder should be initialized"

            if user_cache:
                workflow = await user_cache.get_or_create(
                    cache_key,
                    lambda: graph_builder.build_workflow(user_id),
                )
            else:
                workflow = await graph_builder.build_workflow(user_id)

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

        conversation = await storage.get_service(storage.conversation).get_conversation(
            conversation_id
        )

        summaries = await storage.get_service(
            storage.summary
        ).get_summaries_for_conversation(conversation_id)

        # WorkflowState expects Message objects, not BaseMessage objects
        # So we use the messages directly without LangChain conversion

        current_user_message = next(
            (msg for msg in reversed(messages) if msg.role == MessageRole.USER),
            Message(
                content=[
                    MessageContent(type=MessageContentType.TEXT, text="", url=None)
                ],
                role=MessageRole.USER,
            ),
        )

        # Load active todos for continuation context
        active_todos = await storage.get_service(
            storage.todo
        ).get_todos_by_conversation(user_id, conversation_id)

        # Create the state with centralized user configuration and todo context
        state = WorkflowState(
            title=(
                conversation.title
                if (
                    conversation
                    and not conversation.title.startswith("New conversation")
                )
                else None
            ),
            messages=messages,  # Use Message objects directly
            summaries=summaries,
            current_user_message=current_user_message,  # Use Message object directly
            user_id=user_id,
            user_config=user_config,
            conversation_id=conversation_id,
            active_todos=active_todos,  # Include active todos for context continuity
            dynamic_tool_storage=storage.get_service(
                storage.dynamic_tool
            ),  # Add dynamic tool storage
            checkpoint_metadata={
                "conversation_id": conversation_id,
                "user_id": user_id,
                "turn_timestamp": datetime.now(timezone.utc).isoformat(),
            },
            things_to_remember=[current_user_message],
        )

        return state

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
