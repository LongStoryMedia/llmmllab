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

from models import IntentAnalysis, Message, WorkflowType, LangChainMessage

# Lazy import to avoid circular dependencies
# from db import storage

from composer.graph.state import WorkflowState, ChatWorkflowState, ResearchWorkflowState
from composer.graph.builder import GraphBuilder
from composer.tools.registry import ToolRegistry
from composer.graph.cache import WorkflowCache
from composer.agents.intent_classifier import IntentClassifierAgent
from composer.monitoring.logging import composer_logger


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

        # Import pipeline factory to inject into GraphBuilder
        try:
            from runner import pipeline_factory

            self.pipeline_factory = pipeline_factory
        except ImportError as e:
            self.logger.warning(f"Could not import pipeline_factory: {e}")
            self.pipeline_factory = None

        self.graph_builder = GraphBuilder(pipeline_factory=self.pipeline_factory)
        self.tool_registry = ToolRegistry()
        # Workflow cache is now created per-user during workflow composition
        self.workflow_caches: Dict[str, WorkflowCache] = (
            {}
        )  # Dict[str, WorkflowCache] - keyed by user_id
        self.intent_classifier = IntentClassifierAgent()

        # Initialize core components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize composer components and validate configuration."""
        self.logger.info(
            "ComposerService initialized",
            extra={
                "graph_builder": "ready",
                "tool_registry": "ready",
                "intent_classifier": "ready",
                "workflow_caches": "ready",
            },
        )

    async def compose_workflow(
        self,
        user_id: str,
        messages: List[Message],
        workflow_type: WorkflowType,
    ) -> CompiledStateGraph:
        """
        Construct or retrieve a compiled graph.

        This is the main entry point for workflow composition.

        args:
            user_id: User ID for configuration retrieval
            messages: Conversation messages
            workflow_type: Type of workflow to compose (e.g. "CHAT", "RESEARCH")

        returns:
            CompiledStateGraph: Ready to execute LangGraph workflow
        """
        try:
            # 1. Get user configuration from shared data layer
            # Configuration overrides and defaults are resolved at the data layer
            # ComposerService receives final resolved configuration
            from db import storage  # pylint: disable=import-outside-toplevel

            user_config = await storage.get_service(
                storage.user_config
            ).get_user_config(user_id)

            # 2. Analyze intent before building workflow
            intent = await self._analyze_intent(user_id, messages)

            # 3. Get tools for this context and intent
            tools = await self.tool_registry.get_tools_for_context(intent, user_id)

            # 4. Use per-user cache if enabled
            user_cache = None
            if user_config.workflow.enable_workflow_caching:
                if user_id not in self.workflow_caches:
                    self.workflow_caches[user_id] = WorkflowCache()
                user_cache = self.workflow_caches[user_id]

                cache_key = await user_cache.get_cache_key(
                    user_id, workflow_type, tools
                )

                cached_workflow = await user_cache.get(cache_key)
                if cached_workflow:
                    self.logger.debug(
                        "Retrieved workflow from cache", extra={"cache_key": cache_key}
                    )
                    return cached_workflow

            # 5. Build new workflow (configuration retrieved internally from data layer)
            # Note: Type conversion handled by ToolRegistry.get_tools_for_context
            # which returns List[AvailableTool] compatible objects
            builder_fn = lambda: self.graph_builder.build_from_context(
                user_id, messages, tools, workflow_type
            )

            if user_cache:
                workflow = await user_cache.get_or_create(cache_key, builder_fn)
            else:
                workflow = await builder_fn()

            self.logger.info(
                "Workflow composed successfully",
                extra={
                    "workflow_type": workflow_type,
                    "tool_count": len(tools),
                    "user_id": user_id,
                },
            )

            return workflow

        except Exception as e:
            self.logger.error(
                "Failed to compose workflow",
                extra={
                    "workflow_type": workflow_type,
                    "error": str(e),
                    "user_id": user_id,
                },
                exc_info=True,
            )
            raise

    async def _analyze_intent(
        self, user_id: str, messages: List[Message]
    ) -> "IntentAnalysis":
        """
        Use an LLM-based intent agent to analyze conversation context.

        This analysis guides tool selection, workflow type selection,
        and RAG depth configuration.
        """
        try:
            return await self.intent_classifier.analyze(user_id, messages)
        except Exception as e:
            self.logger.warning(
                "Intent analysis failed, using defaults",
                extra={"error": str(e), "user_id": user_id},
                exc_info=True,
            )
            raise

    async def create_initial_state(
        self,
        user_id: str,
        messages: List[Message],
        workflow_type: WorkflowType,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowState:
        """Create initial workflow state from user configuration."""

        # Get user configuration for workflow preferences
        from db import storage  # pylint: disable=import-outside-toplevel

        user_config = await storage.get_service(storage.user_config).get_user_config(
            user_id
        )

        # Choose appropriate state class based on workflow type
        if workflow_type == WorkflowType.RESEARCH:
            state_class = ResearchWorkflowState
        elif workflow_type == WorkflowType.CHAT:
            state_class = ChatWorkflowState
        else:
            state_class = WorkflowState

        langchain_messages = []
        for msg in messages:
            if hasattr(msg, "content") and hasattr(msg, "role"):
                # Convert from Message to LangChainMessage
                # Extract text content from MessageContent list
                content_text = ""
                if isinstance(msg.content, list):
                    content_parts = []
                    for content_part in msg.content:
                        if hasattr(content_part, "text"):
                            content_parts.append(content_part.text)
                        elif isinstance(content_part, str):
                            content_parts.append(content_part)
                    content_text = "\n".join(content_parts)
                else:
                    content_text = str(msg.content)

                langchain_messages.append(
                    LangChainMessage(
                        content=content_text,
                        type="human" if msg.role.value == "user" else "ai",
                    )
                )
            else:
                langchain_messages.append(msg)  # Assume already correct format

        state = state_class(
            messages=langchain_messages,
            user_id=user_id,
            workflow_type=workflow_type,
            execution_metadata={
                "created_at": asyncio.get_event_loop().time(),
                "composer_version": "0.1.0",
                # Include user workflow preferences in metadata
                "streaming_enabled": user_config.workflow.enable_streaming,
                "workflow_timeout": user_config.workflow.default_timeout,
            },
        )

        # Add additional context
        if additional_context:
            state.execution_metadata.update(additional_context)

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
            streaming_enabled = initial_state.execution_metadata.get(
                "streaming_enabled", True  # Default to True if not specified
            )
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

        await self.tool_registry.close()

        # Close other resources as needed
