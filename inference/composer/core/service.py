"""
Main ComposerService orchestrator.
Central to the redesign - serves as the primary, authoritative execution runtime.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from models.conversation_ctx import ConversationCtx
from composer.graph.state import WorkflowState, ChatWorkflowState, ResearchWorkflowState
from composer.graph.builder import GraphBuilder
from composer.tools.registry import ToolRegistry
from composer.graph.cache import WorkflowCache
from composer.agents.intent_classifier import IntentClassifierAgent
from composer.config import config


@dataclass
class WorkflowType:
    """Enumeration of supported workflow types."""

    CHAT = "CHAT"
    RESEARCH = "RESEARCH"
    MULTI_AGENT = "MULTI_AGENT"
    CREATIVE = "CREATIVE"


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
        self.logger = logging.getLogger(__name__)
        self.graph_builder = GraphBuilder()
        self.tool_registry = ToolRegistry()
        # Initialize workflow cache if enabled
        self.workflow_cache = (
            WorkflowCache() if config.default_workflow.enable_workflow_caching else None
        )
        self.intent_classifier = IntentClassifierAgent()

        # Initialize core components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize composer components and validate configuration."""
        composer_logger.logger.info(
            "Service initialized",
            extra={
                "caching_enabled": config.default_workflow.enable_workflow_caching,
                "streaming_enabled": config.default_workflow.enable_streaming,
                "multi_agent_enabled": config.default_workflow.enable_multi_agent,
            },
        )

    async def compose_workflow(
        self,
        conversation_ctx: ConversationCtx,
        workflow_type: str,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> "CompiledGraph":
        """
        Construct or retrieve a compiled graph for the given conversation.

        This is the main entry point for workflow composition, implementing
        the core architectural shift to Composer-centric design.
        """
        try:
            # 1. Analyze intent before building workflow
            intent = await self._analyze_intent(conversation_ctx)

            # 2. Get tools for this context and intent
            tools = await self.tool_registry.get_tools_for_context(
                intent, conversation_ctx
            )

            # 3. Merge configuration overrides
            workflow_config = self._merge_config_overrides(
                conversation_ctx, workflow_type, intent, config_overrides
            )

            # 4. Use cache if available
            if self.workflow_cache:
                cache_key = self.workflow_cache.get_cache_key(
                    conversation_ctx.user_config, workflow_type, tools
                )

                cached_workflow = await self.workflow_cache.get(cache_key)
                if cached_workflow:
                    self.logger.debug(
                        "Retrieved workflow from cache", extra={"cache_key": cache_key}
                    )
                    return cached_workflow

            # 5. Build new workflow
            builder_fn = lambda: self.graph_builder.build_from_context(
                conversation_ctx, tools, workflow_config, workflow_type
            )

            if self.workflow_cache:
                workflow = await self.workflow_cache.get_or_create(
                    cache_key, builder_fn
                )
            else:
                workflow = await builder_fn()

            self.logger.info(
                "Workflow composed successfully",
                extra={
                    "workflow_type": workflow_type,
                    "tool_count": len(tools),
                    "user_id": (
                        conversation_ctx.user_config.user_id
                        if conversation_ctx.user_config
                        else None
                    ),
                },
            )

            return workflow

        except Exception as e:
            self.logger.error(
                "Failed to compose workflow",
                extra={
                    "workflow_type": workflow_type,
                    "error": str(e),
                    "user_id": (
                        conversation_ctx.user_config.user_id
                        if conversation_ctx.user_config
                        else None
                    ),
                },
                exc_info=True,
            )
            raise

    async def _analyze_intent(
        self, conversation_ctx: ConversationCtx
    ) -> "IntentAnalysis":
        """
        Use an LLM-based intent agent to analyze conversation context.

        This analysis guides tool selection, workflow type selection,
        and RAG depth configuration.
        """
        try:
            return await self.intent_classifier.analyze(conversation_ctx)
        except Exception as e:
            self.logger.warning(
                "Intent analysis failed, using defaults",
                extra={"error": str(e)},
                exc_info=True,
            )
            # Return default intent analysis
            from models.intent_analysis import IntentAnalysis

            return IntentAnalysis(
                primary_intent="chat",
                confidence=0.5,
                requires_tools=False,
                estimated_complexity="low",
            )

    def _merge_config_overrides(
        self,
        conversation_ctx: ConversationCtx,
        workflow_type: str,
        intent: "IntentAnalysis",
        config_overrides: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merge user configuration with workflow-specific overrides."""
        # User config always has workflow and tool configs with proper defaults from storage layer
        if not conversation_ctx.user_config:
            raise ValueError("ConversationCtx must have user_config with workflow and tool configs")
        
        workflow_config = conversation_ctx.user_config.workflow
        tool_config = conversation_ctx.user_config.tool

        base_config = {
            "streaming_enabled": workflow_config.enable_streaming,
            "timeout": workflow_config.default_timeout,
            "max_context_length": workflow_config.max_context_length,
            "enable_multi_agent": workflow_config.enable_multi_agent,
            "max_parallel_tools": workflow_config.max_parallel_tools,
            "tool_similarity_threshold": tool_config.tool_similarity_threshold,
            "enable_tool_generation": tool_config.enable_tool_generation,
            "tool_timeout": tool_config.tool_timeout,
        }

        # Add user-specific configuration if available
        if conversation_ctx.user_config:
            base_config.update(
                {
                    "user_preferences": conversation_ctx.user_config.preferences,
                    "model_profiles": conversation_ctx.user_config.model_profiles,
                }
            )

        # Add intent-specific configuration
        if intent:
            base_config.update(
                {
                    "intent_type": intent.primary_intent,
                    "requires_tools": hasattr(intent, "requires_tools")
                    and intent.requires_tools,
                }
            )

        # Apply any additional overrides
        if config_overrides:
            base_config.update(config_overrides)

        return base_config

    async def create_initial_state(
        self,
        conversation_ctx: ConversationCtx,
        workflow_type: str,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowState:
        """Create initial workflow state from conversation context."""

        # Choose appropriate state class based on workflow type
        if workflow_type == WorkflowType.RESEARCH:
            state_class = ResearchWorkflowState
        elif workflow_type == WorkflowType.CHAT:
            state_class = ChatWorkflowState
        else:
            state_class = WorkflowState

        # Extract messages from conversation
        messages = []
        if conversation_ctx.messages:
            messages = conversation_ctx.messages

        # Create base state with user workflow configuration
        state = state_class(
            messages=messages,
            user_id=(
                conversation_ctx.user_config.user_id
                if conversation_ctx.user_config
                else None
            ),
            conversation_id=getattr(conversation_ctx, "conversation_id", None),
            workflow_type=workflow_type,
            execution_metadata={
                "created_at": asyncio.get_event_loop().time(),
                "composer_version": "0.1.0",
                # Include user workflow preferences in metadata
                "streaming_enabled": conversation_ctx.user_config.workflow.enable_streaming if conversation_ctx.user_config else True,
                "workflow_timeout": conversation_ctx.user_config.workflow.default_timeout if conversation_ctx.user_config else 60.0,
            },
        )

        # Add additional context
        if additional_context:
            state.execution_metadata.update(additional_context)

        return state

    async def execute_workflow(
        self,
        workflow: "CompiledGraph",
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
                    initial_state.dict(), version="v2"
                ):
                    yield event
            else:
                # Batch execution
                result = await workflow.ainvoke(initial_state.dict())
                yield {"event": "workflow_complete", "data": result}

        except Exception as e:
            self.logger.error(
                "Workflow execution failed", extra={"error": str(e)}, exc_info=True
            )
            yield {"event": "workflow_error", "data": {"error": str(e)}}

    async def shutdown(self):
        """Clean up resources on service shutdown."""
        self.logger.info("Shutting down ComposerService")

        if self.workflow_cache:
            await self.workflow_cache.close()

        await self.tool_registry.close()

        # Close other resources as needed
