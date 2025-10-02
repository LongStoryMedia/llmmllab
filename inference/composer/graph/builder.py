"""
GraphBuilder for dynamic workflow construction.
Constructs LangGraph workflows dynamically based on conversation context and tools.
"""

import asyncio
from typing import Any, Optional, Dict, List

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy
from langchain_core.runnables import RunnableParallel, RunnableLambda
from models.workflow_type import WorkflowType
from models import ModelProfileType
from models.available_tool import AvailableTool
from composer.monitoring.logging import composer_logger
from composer.core.errors import WorkflowConstructionError
from composer.graph.state import WorkflowState

# Node imports
from composer.nodes.standard import PipelineNode
from composer.nodes.intent_classifier import IntentClassifierNode
from composer.nodes.engineering_agent import EngineeringAgentNode
from composer.nodes.rag.executor import EnhancedRAGExecutor
from composer.tools.registry import ToolRegistry
from composer.workflows.chat import build_chat_workflow
from composer.workflows.research import build_research_workflow as build_research_workflow_impl


class GraphBuilder:
    """
    Constructs LangGraph workflows dynamically based on context.

    The GraphBuilder implements the core workflow construction logic,
    supporting different workflow types with appropriate node compositions.
    """

    def __init__(self, pipeline_factory=None):
        self.pipeline_factory = pipeline_factory
        # Circuit breaker for resilience
        self._circuit_breaker_failures = {}
        self._max_failures = 3
        self._reset_timeout = 60  # seconds

        # Performance optimization: Cache compiled subgraphs
        self._subgraph_cache = {}

        composer_logger.logger.info(
            "GraphBuilder initialized with production optimizations",
            extra={
                "has_pipeline_factory": pipeline_factory is not None,
                "circuit_breaker_enabled": True,
                "subgraph_caching_enabled": True,
            },
        )

    async def _get_user_config(self, user_id: str):
        """Get user configuration from shared data layer."""
        try:
            # Database import - lazy loaded to avoid circular dependencies
            from db import storage  # pylint: disable=import-outside-toplevel

            # Initialize storage if not done
            if not storage.pool:
                composer_logger.logger.warning(
                    "Database not initialized for GraphBuilder"
                )
                return None

            user_config = await storage.get_service(
                storage.user_config
            ).get_user_config(user_id)
            if not user_config:
                composer_logger.logger.warning(
                    f"No user config found for {user_id} in GraphBuilder"
                )
                return None
            return user_config
        except Exception as e:
            composer_logger.logger.error(
                f"Failed to get user config for {user_id} in GraphBuilder: {e}"
            )
            return None

    async def _get_workflow_tools(self, user_id: str) -> List[AvailableTool]:
        """Get available tools for a workflow, using simplified collection for dedicated workflows."""
        try:
            registry = ToolRegistry()
            
            # Get basic tool set - static tools primarily
            tools = []
            
            # Add static tools
            for name, tool_cls in registry.static_tools.items():
                if tool_cls:
                    available_tool = AvailableTool(
                        name=name, 
                        description=f"Static tool: {name}", 
                        type="static"
                    )
                    tools.append(available_tool)
            
            # Add a few dynamic tools if user_id provided
            if user_id:
                for tool_id, tool_instance in list(registry.dynamic_tools.items())[:3]:  # Limit for performance
                    if tool_instance:
                        available_tool = AvailableTool(
                            name=f"dynamic_{tool_id}",
                            description=f"Dynamic tool: {tool_id}",
                            type="dynamic",
                        )
                        tools.append(available_tool)
                        
            return tools
            
        except Exception as e:
            composer_logger.logger.error(
                f"Failed to get workflow tools for {user_id}: {e}"
            )
            return []  # Return empty list on error

    async def build_from_context(
        self, user_id: str, workflow_type: WorkflowType
    ) -> CompiledStateGraph:
        """
        Build workflow from user configuration, tools, and workflow type.

        This is the main entry point for dynamic workflow construction.
        Configuration is retrieved from shared data layer using user_id.
        """
        try:
            composer_logger.logger.info(
                "Building workflow from context",
                extra={
                    "workflow_type": workflow_type,
                    "user_id": user_id,
                },
            )

            # Select appropriate build method based on workflow type
            if workflow_type == WorkflowType.CHAT:
                return await self._build_chat_subgraph()
            elif workflow_type == WorkflowType.RESEARCH:
                return await self.build_research_workflow(user_id)
            elif workflow_type == WorkflowType.MULTI_AGENT:
                return await self.build_multi_agent_workflow(user_id)
            elif workflow_type == WorkflowType.CREATIVE:
                return await self.build_creative_workflow(user_id)
            else:
                # Default to chat workflow
                return await self._build_chat_subgraph()

        except Exception as e:
            composer_logger.log_error(
                e, {"context": "workflow_construction", "workflow_type": workflow_type}
            )
            raise WorkflowConstructionError(
                f"Failed to build {workflow_type} workflow: {e}"
            ) from e

    async def build_master_workflow(
        self, user_id: str, workflow_type: Optional[WorkflowType] = None
    ) -> CompiledStateGraph:
        """
        Build master workflow with intelligent routing to subgraphs.

        Creates a single graph with:
        1. Intent analysis node (unless explicit workflow_type provided)
        2. Router node that determines execution strategy
        3. Conditional routing to appropriate subgraph(s)
        4. Support for parallel, series, or single execution strategies

        Args:
            user_id: User ID for configuration retrieval
            workflow_type: Optional explicit workflow type for direct routing

        Returns:
            CompiledStateGraph: Master workflow with intelligent subgraph routing
        """
        try:
            composer_logger.logger.info(
                "Building master workflow with subgraph routing",
                extra={"user_id": user_id, "workflow_type": workflow_type},
            )

            # Create master workflow graph
            workflow = StateGraph(WorkflowState)

            # Add intent analysis node (always present for context enrichment)
            workflow.add_node("intent_analysis", IntentClassifierNode())

            # Add tool collection node (collects available tools based on intent)
            workflow.add_node(
                "tool_collection", self._create_tool_collection_node(user_id)
            )

            # Enhanced execution pattern: Use direct execution for simple workflows,
            # subgraph routing for complex scenarios or when workflow_type is not specified
            use_enhanced_execution = workflow_type is not None and workflow_type in [
                WorkflowType.CHAT,
                WorkflowType.RESEARCH,
                WorkflowType.CREATIVE,
            ]

            if use_enhanced_execution:
                # Enhanced direct execution pattern (from enhanced_builder)
                # workflow_type is guaranteed to be non-None here due to use_enhanced_execution check
                assert workflow_type is not None  # Type narrowing for mypy
                executor = self._create_workflow_executor(user_id, workflow_type)
                workflow.add_node("enhanced_executor", executor)

                # Simple direct routing for enhanced execution
                workflow.set_entry_point("intent_analysis")
                workflow.add_edge("intent_analysis", "tool_collection")
                workflow.add_edge("tool_collection", "enhanced_executor")
                workflow.add_edge("enhanced_executor", END)

            else:
                # Complex subgraph routing (preserve existing comprehensive capabilities)
                # Add router node with routing logic
                workflow.add_node(
                    "router", self._create_router_node(user_id, workflow_type)
                )

                # Create and add subgraphs as compiled nodes
                subgraphs = await self._create_all_subgraphs(user_id)

                # Add subgraph nodes
                for name, subgraph in subgraphs.items():
                    workflow.add_node(f"{name}_subgraph", subgraph)

                # Add execution coordinator node
                workflow.add_node("coordinator", self._create_coordinator_node(user_id))

                # Define complex workflow edges
                workflow.set_entry_point("intent_analysis")
                workflow.add_edge("intent_analysis", "tool_collection")
                workflow.add_edge("tool_collection", "router")

                # Conditional routing from router to subgraphs
                workflow.add_conditional_edges(
                    "router",
                    self._route_to_subgraphs,
                    {
                        "chat": "chat_subgraph",
                        "research": "research_subgraph",
                        "creative": "creative_subgraph",
                        "multi_agent": "multi_agent_subgraph",
                        "coordinator": "coordinator",  # For parallel/series execution
                    },
                )

                # All subgraphs route to coordinator for result processing
                for name in subgraphs.keys():
                    workflow.add_edge(f"{name}_subgraph", "coordinator")

                workflow.add_edge("coordinator", END)

            # Compile and return the master workflow
            compiled_workflow = workflow.compile()

            if use_enhanced_execution:
                composer_logger.logger.info(
                    "Enhanced master workflow compiled successfully",
                    extra={
                        "user_id": user_id,
                        "execution_pattern": "enhanced_direct",
                        "workflow_type": workflow_type.value if workflow_type else None,
                    },
                )
            else:
                # For complex routing, we know subgraphs was created
                subgraph_count = len(
                    [n for n in workflow.nodes.keys() if n.endswith("_subgraph")]
                )
                composer_logger.logger.info(
                    "Master workflow with subgraphs compiled successfully",
                    extra={
                        "user_id": user_id,
                        "execution_pattern": "complex_subgraphs",
                        "subgraph_count": subgraph_count,
                    },
                )

            return compiled_workflow

        except Exception as e:
            composer_logger.logger.error(
                "Failed to build master workflow",
                extra={"user_id": user_id, "error": str(e)},
            )
            # Create simple fallback workflow
            return await self._create_fallback_workflow(user_id)

    async def build_research_workflow(self, user_id: str) -> CompiledStateGraph:
        """
        Build a research workflow using dedicated workflow implementation.
        
        This method delegates to the canonical research workflow definition
        to eliminate code duplication.
        """
        try:
            # Get available tools for this user/workflow
            tools = await self._get_workflow_tools(user_id)
            
            # Use the dedicated research workflow function
            return await build_research_workflow_impl(
                user_id=user_id,
                tools=tools,
                pipeline_factory=self.pipeline_factory
            )
            
        except Exception as e:
            composer_logger.logger.error(
                "Failed to build research workflow",
                extra={"user_id": user_id, "error": str(e)},
            )
            raise WorkflowConstructionError(
                f"Research workflow construction failed: {e}"
            ) from e

    async def build_multi_agent_workflow(self, user_id: str) -> CompiledStateGraph:
        """
        Build multi-agent orchestration workflow.

        Workflow: Agent Router -> Specialized Agents -> Coordination -> Final Response
        Configuration retrieved from shared data layer using user_id.
        """
        # Get user configuration from shared data layer
        user_config = await self._get_user_config(user_id)
        workflow_config_obj = user_config.workflow if user_config else None

        nodes = [
            "agent_router",
            "specialist_agent_1",
            "specialist_agent_2",
            "coordination",
            "final_response",
        ]

        try:
            composer_logger.logger.info(
                "Building multi-agent workflow",
                extra={"user_id": user_id},
            )

            # Create workflow graph
            workflow = StateGraph(WorkflowState)

            # Configuration retrieved internally from shared data layer using user_id
            user_config = await self._get_user_config(user_id)
            if not user_config:
                raise WorkflowConstructionError("Unable to retrieve user configuration")

            # Use injected pipeline factory
            pipeline_factory = self.pipeline_factory

            # Add multi-agent coordination nodes
            workflow.add_node("agent_router", IntentClassifierNode())
            workflow.add_node(
                "specialist_agent_1", EngineeringAgentNode(pipeline_factory)
            )
            workflow.add_node(
                "specialist_agent_2",
                PipelineNode(pipeline_factory, ModelProfileType.Analysis, stream=True),
            )
            workflow.add_node(
                "coordination",
                PipelineNode(pipeline_factory, ModelProfileType.Primary, stream=True),
            )
            workflow.add_node(
                "final_response",
                PipelineNode(pipeline_factory, ModelProfileType.Primary, stream=True),
            )

            # Define multi-agent workflow logic
            workflow.set_entry_point("agent_router")
            workflow.add_edge("agent_router", "specialist_agent_1")
            workflow.add_edge("specialist_agent_1", "coordination")
            workflow.add_edge("coordination", "final_response")
            workflow.add_edge("final_response", END)

            # Compile workflow
            compiled_workflow = workflow.compile()

            composer_logger.logger.info(
                "Built multi-agent workflow",
                extra={
                    "node_count": len(nodes),
                    "enable_handoffs": (
                        workflow_config_obj.enable_multi_agent
                        if workflow_config_obj
                        else True
                    ),
                },
            )

            return compiled_workflow

        except Exception as e:
            composer_logger.logger.error(
                "Failed to build multi-agent workflow",
                extra={"user_id": user_id, "error": str(e)},
            )
            raise WorkflowConstructionError(
                f"Multi-agent workflow construction failed: {e}"
            ) from e

    async def build_creative_workflow(self, user_id: str) -> CompiledStateGraph:
        """
        Build creative content generation workflow.

        Workflow: Creative Planning -> Content Generation -> Refinement -> Output
        Configuration retrieved from shared data layer using user_id.
        """
        # Get user configuration from shared data layer
        user_config = await self._get_user_config(user_id)
        refinement_config = user_config.refinement if user_config else None

        nodes = [
            "creative_planning",
            "content_generation",
            "refinement",
            "output_formatting",
        ]

        try:
            composer_logger.logger.info(
                "Building creative workflow",
                extra={"user_id": user_id},
            )

            # Create workflow graph
            workflow = StateGraph(WorkflowState)

            # Configuration retrieved internally from shared data layer using user_id
            user_config = await self._get_user_config(user_id)
            if not user_config:
                raise WorkflowConstructionError("Unable to retrieve user configuration")

            # Use injected pipeline factory
            pipeline_factory = self.pipeline_factory

            # Add creative workflow nodes
            workflow.add_node(
                "creative_planning",
                PipelineNode(pipeline_factory, ModelProfileType.Primary, stream=False),
            )
            workflow.add_node(
                "content_generation",
                PipelineNode(pipeline_factory, ModelProfileType.Primary, stream=True),
            )
            workflow.add_node(
                "refinement",
                PipelineNode(
                    pipeline_factory, ModelProfileType.SelfCritique, stream=False
                ),
            )
            workflow.add_node(
                "output_formatting",
                PipelineNode(
                    pipeline_factory, ModelProfileType.Formatting, stream=True
                ),
            )

            # Define creative workflow logic
            workflow.set_entry_point("creative_planning")
            workflow.add_edge("creative_planning", "content_generation")
            workflow.add_edge("content_generation", "refinement")
            workflow.add_edge("refinement", "output_formatting")
            workflow.add_edge("output_formatting", END)

            # Compile workflow
            compiled_workflow = workflow.compile()

            composer_logger.logger.info(
                "Built creative workflow",
                extra={
                    "node_count": len(nodes),
                    "enable_critique": (
                        refinement_config.enable_response_critique
                        if refinement_config
                        else True
                    ),
                },
            )

            return compiled_workflow

        except Exception as e:
            composer_logger.logger.error(
                "Failed to build creative workflow",
                extra={"user_id": user_id, "error": str(e)},
            )
            raise WorkflowConstructionError(
                f"Creative workflow construction failed: {e}"
            ) from e

    async def _create_all_subgraphs(self, user_id: str) -> dict:
        """Create all workflow subgraphs."""
        try:
            subgraphs = {
                "chat": await self._build_chat_subgraph(),
                "research": await self.build_research_workflow(user_id),
                "creative": await self.build_creative_workflow(user_id),
                "multi_agent": await self.build_multi_agent_workflow(user_id),
            }
            return subgraphs
        except Exception as e:
            composer_logger.logger.error(f"Failed to create subgraphs: {e}")
            # Return minimal chat subgraph as fallback
            return {"chat": await self._build_chat_subgraph()}

    async def _build_chat_subgraph(self) -> CompiledStateGraph:
        """Build chat workflow using dedicated workflow implementation."""
        try:
            # Get available tools for chat workflow
            tools = await self._get_workflow_tools("")  # Empty user_id for subgraph
            
            # Use the dedicated chat workflow function
            return await build_chat_workflow(
                user_id="",  # Empty for subgraph usage
                tools=tools,
                pipeline_factory=self.pipeline_factory
            )
            
        except Exception as e:
            composer_logger.logger.error(f"Failed to build chat subgraph: {e}")
            # Create minimal fallback on error
            workflow = StateGraph(WorkflowState)
            workflow.add_node(
                "chat_response",
                PipelineNode(
                    self.pipeline_factory,
                    ModelProfileType.Primary,
                    stream=True,
                ),
            )
            workflow.set_entry_point("chat_response")
            workflow.add_edge("chat_response", END)
            return workflow.compile()

    def _create_coordinator_node(self, user_id: str):
        """Create coordinator node for handling execution strategy."""

        async def coordinate_execution(state):
            """Coordinate execution of multiple subgraphs based on strategy."""
            try:
                execution_strategy = getattr(state, "execution_strategy", "single")
                selected_workflows = getattr(state, "selected_workflows", ["chat"])

                composer_logger.logger.info(
                    "Coordinating workflow execution",
                    extra={
                        "user_id": user_id,
                        "strategy": execution_strategy,
                        "workflows": selected_workflows,
                    },
                )

                # For now, just pass through - coordination logic can be enhanced
                state.final_response = getattr(state, "response", "Processing complete")
                return state

            except Exception as e:
                composer_logger.logger.error(
                    "Coordination failed", extra={"user_id": user_id, "error": str(e)}
                )
                state.final_response = (
                    "Sorry, there was an error processing your request."
                )
                return state

        return coordinate_execution

    def _route_to_subgraphs(self, state: WorkflowState) -> str:
        """Determine which subgraph to route to using Command primitive for deterministic routing."""
        try:
            # Check for Command-based routing decision (production-grade deterministic routing)
            if state.next_node:
                composer_logger.logger.info(
                    "Using Command-based deterministic routing",
                    extra={"next_node": state.next_node},
                )
                return state.next_node

            # Check for explicit routing decision from router (strongly typed)
            if state.routing_decision:
                composer_logger.logger.info(
                    "Using explicit routing decision",
                    extra={"routing_decision": state.routing_decision.value},
                )
                return state.routing_decision.value

            # Fallback to intent-based routing with structured validation
            intent_analysis = getattr(state, "intent_classification", None)
            if intent_analysis and hasattr(intent_analysis, "primary_intent"):
                primary_intent = intent_analysis.primary_intent.lower()
                # Use deterministic mapping instead of string matching
                routing_map = {
                    "research": "research",
                    "analysis": "research",
                    "creative": "creative",
                    "generate": "creative",
                    "multi_agent": "multi_agent",
                    "collaboration": "multi_agent",
                }

                for key, route in routing_map.items():
                    if key in primary_intent:
                        return route

            # Default fallback
            return "chat"

        except Exception as e:
            composer_logger.logger.error(
                "Routing error - falling back to chat", extra={"error": str(e)}
            )
            return "chat"

    async def _create_fallback_workflow(
        self, user_id: str
    ) -> CompiledStateGraph:  # noqa: ARG002
        """Create minimal fallback workflow."""
        try:
            return await self._build_chat_subgraph()
        except Exception as e:
            composer_logger.logger.error(f"Fallback workflow creation failed: {e}")
            # Create absolute minimal workflow
            workflow = StateGraph(WorkflowState)

            async def minimal_response(state):
                state.response = "I apologize, but I'm experiencing technical difficulties. Please try again later."
                return state

            workflow.add_node("minimal_response", minimal_response)
            workflow.set_entry_point("minimal_response")
            workflow.add_edge("minimal_response", END)

            return workflow.compile()

    def _create_router_node(
        self, user_id: str, explicit_workflow_type: Optional[WorkflowType] = None
    ):
        """
        Create intelligent router node that determines workflow execution strategy.

        The router can decide to:
        - Route to single subgraph based on intent or explicit type
        - Execute multiple subgraphs in parallel
        - Execute subgraphs in series
        - Use hybrid parallel+series execution
        """

        async def route_workflows(state):
            """Route to appropriate subgraph(s) based on intent analysis or explicit type."""
            try:
                if explicit_workflow_type:
                    # Explicit routing - force specific workflow type
                    state.selected_workflows = [explicit_workflow_type.value]
                    state.execution_strategy = "single"
                    return state

                # Intelligent routing based on intent analysis
                intent = getattr(state, "intent", "chat")
                complexity = getattr(state, "complexity", "simple")

                # Route based on intent and complexity
                if complexity == "high" and "research" in intent.lower():
                    # High complexity research might need multiple workflows
                    state.selected_workflows = ["research", "creative"]
                    state.execution_strategy = (
                        "series"  # Research first, then creative synthesis
                    )
                elif "multi" in intent.lower() or "agent" in intent.lower():
                    state.selected_workflows = ["multi_agent"]
                    state.execution_strategy = "single"
                elif "creative" in intent.lower() or "generate" in intent.lower():
                    state.selected_workflows = ["creative"]
                    state.execution_strategy = "single"
                elif "research" in intent.lower() or "analyze" in intent.lower():
                    state.selected_workflows = ["research"]
                    state.execution_strategy = "single"
                else:
                    # Default to chat for simple interactions
                    state.selected_workflows = ["chat"]
                    state.execution_strategy = "single"

                composer_logger.logger.info(
                    "Router determined execution strategy",
                    extra={
                        "user_id": user_id,
                        "intent": intent,
                        "complexity": complexity,
                        "selected_workflows": state.selected_workflows,
                        "execution_strategy": state.execution_strategy,
                    },
                )

                return state

            except Exception as e:
                composer_logger.logger.error(
                    "Router failed, falling back to chat",
                    extra={"user_id": user_id, "error": str(e)},
                )
                # Fallback to chat workflow
                state.selected_workflows = ["chat"]
                state.execution_strategy = "single"
                return state

        return route_workflows

    def _create_tool_collection_node(self, user_id: str):
        """
        Create optimized tool collection node with LCEL parallel execution.

        Implements intra-node concurrency for maximum performance during tool discovery and collection.
        Uses ToolRegistry with parallel static/dynamic tool collection.
        """

        async def collect_tools_parallel(state):
            """Optimized tool collection with parallel execution for performance."""
            try:
                # Circuit breaker check
                if self._is_circuit_open("tool_collection"):
                    composer_logger.logger.warning(
                        "Circuit breaker open for tool_collection",
                        extra={"user_id": user_id},
                    )
                    # Return cached/minimal tools
                    state.required_tools = []
                    return state

                # LCEL RunnableParallel for concurrent tool operations
                parallel_operations = RunnableParallel(
                    {
                        "static_tools": RunnableLambda(self._collect_static_tools),
                        "dynamic_tools": RunnableLambda(self._collect_dynamic_tools),
                        "intent_tools": RunnableLambda(
                            self._collect_intent_based_tools
                        ),
                    }
                )

                # Execute tool collection operations concurrently with strongly typed state
                tool_results = await parallel_operations.ainvoke(
                    {
                        "state": state,
                        "user_id": user_id,
                        "intent": state.intent_classification,  # Use strongly typed field
                    }
                )

                # Merge results efficiently
                all_tools = []
                for tool_list in tool_results.values():
                    if tool_list:
                        all_tools.extend(tool_list)

                # Deduplicate and optimize tool list
                unique_tools = self._deduplicate_tools(all_tools)
                state.required_tools = unique_tools

                composer_logger.logger.info(
                    "Parallel tool collection completed",
                    extra={
                        "user_id": user_id,
                        "total_tools": len(unique_tools),
                        "static_count": len(tool_results.get("static_tools", [])),
                        "dynamic_count": len(tool_results.get("dynamic_tools", [])),
                        "intent_count": len(tool_results.get("intent_tools", [])),
                    },
                )

                return state

            except Exception as e:
                # Circuit breaker increment
                self._record_failure("tool_collection")

                composer_logger.logger.error(
                    "Tool collection failed",
                    extra={"user_id": user_id, "error": str(e)},
                )
                # Graceful degradation - return minimal toolset
                state.required_tools = []
                return state

        return collect_tools_parallel

    def _is_circuit_open(self, operation: str) -> bool:
        """Check if circuit breaker is open for given operation."""
        if operation not in self._circuit_breaker_failures:
            return False

        failures, last_failure = self._circuit_breaker_failures[operation]

        # Reset circuit if timeout exceeded
        if (asyncio.get_event_loop().time() - last_failure) > self._reset_timeout:
            del self._circuit_breaker_failures[operation]
            return False

        return failures >= self._max_failures

    def _record_failure(self, operation: str):
        """Record failure for circuit breaker."""
        current_time = asyncio.get_event_loop().time()
        if operation in self._circuit_breaker_failures:
            failures, _ = self._circuit_breaker_failures[operation]
            self._circuit_breaker_failures[operation] = (failures + 1, current_time)
        else:
            self._circuit_breaker_failures[operation] = (1, current_time)

    async def _collect_static_tools(
        self, input_data: Dict[str, Any]
    ) -> List[AvailableTool]:
        """Collect static tools with error handling."""
        try:
            registry = ToolRegistry()
            # Initialize static tools using available methods\n            # Note: Static tools are loaded during ToolRegistry init"

            # Convert static tools to AvailableTool format
            static_tools = []
            for name, tool_cls in registry.static_tools.items():
                if tool_cls:
                    available_tool = AvailableTool(
                        name=name, description=f"Static tool: {name}", type="static"
                    )
                    static_tools.append(available_tool)

            composer_logger.logger.debug(
                "Static tools collected",
                extra={"count": len(static_tools), "input_context": bool(input_data)},
            )

            return static_tools
        except Exception as e:
            composer_logger.logger.error(f"Static tool collection failed: {e}")
            return []

    async def _collect_dynamic_tools(
        self, input_data: Dict[str, Any]
    ) -> List[AvailableTool]:
        """Collect dynamic tools with error handling."""
        try:
            registry = ToolRegistry()
            user_id = input_data.get("user_id")

            # Get dynamic tools from registry
            dynamic_tools = []
            for tool_id, tool_instance in registry.dynamic_tools.items():
                if tool_instance:
                    available_tool = AvailableTool(
                        name=f"dynamic_{tool_id}",
                        description=f"Dynamic tool: {tool_id}",
                        type="dynamic",
                    )
                    dynamic_tools.append(available_tool)

            return dynamic_tools if user_id else []
        except Exception as e:
            composer_logger.logger.error(f"Dynamic tool collection failed: {e}")
            return []

    async def _collect_intent_based_tools(
        self, input_data: Dict[str, Any]
    ) -> List[AvailableTool]:
        """Collect tools based on intent analysis using ToolRegistry.get_tools_for_context."""
        try:
            intent = input_data.get("intent")
            user_id = input_data.get("user_id")

            if not intent or not user_id:
                return []

            registry = ToolRegistry()
            # Use the correct method from ToolRegistry
            tools = await registry.get_tools_for_context(intent, user_id)
            return tools if tools else []
        except Exception as e:
            composer_logger.logger.error(f"Intent-based tool collection failed: {e}")
            return []

    def _deduplicate_tools(self, tools: List[AvailableTool]) -> List[AvailableTool]:
        """Deduplicate tools by name, keeping the first occurrence."""
        seen_names = set()
        unique_tools = []

        for tool in tools:
            tool_name = getattr(tool, "name", str(tool))
            if tool_name not in seen_names:
                seen_names.add(tool_name)
                unique_tools.append(tool)

        return unique_tools

    def _get_retry_policy(self, operation_type: str) -> RetryPolicy:
        """Get retry policy based on operation type for resilience."""
        retry_policies = {
            "llm_call": RetryPolicy(
                max_attempts=3, backoff_factor=1.5, initial_interval=1.0
            ),
            "tool_execution": RetryPolicy(
                max_attempts=2, backoff_factor=2.0, initial_interval=0.5
            ),
            "rag_operation": RetryPolicy(
                max_attempts=2, backoff_factor=1.5, initial_interval=2.0
            ),
            "default": RetryPolicy(
                max_attempts=2, backoff_factor=1.0, initial_interval=1.0
            ),
        }

        return retry_policies.get(operation_type, retry_policies["default"])

    # Enhanced Execution Methods (from enhanced_builder pattern)

    def _create_workflow_executor(self, user_id: str, workflow_type: WorkflowType):
        """Create executor for explicit workflow type (enhanced_builder pattern)."""

        async def workflow_executor(state: WorkflowState) -> WorkflowState:
            """Execute specific workflow type with enhanced execution patterns."""
            try:
                composer_logger.logger.info(
                    "Executing explicit workflow",
                    extra={"user_id": user_id, "workflow_type": workflow_type.value},
                )

                if workflow_type == WorkflowType.RESEARCH:
                    return await self._execute_research_flow(state, user_id)
                elif workflow_type == WorkflowType.CREATIVE:
                    return await self._execute_creative_flow(state, user_id)
                elif workflow_type == WorkflowType.MULTI_AGENT:
                    return await self._execute_multi_agent_flow(state, user_id)
                else:  # Default to chat
                    return await self._execute_chat_flow(state, user_id)

            except Exception as e:
                composer_logger.logger.error(
                    "Workflow executor failed",
                    extra={"user_id": user_id, "error": str(e)},
                )
                return await self._execute_chat_flow(state, user_id)  # Fallback

        return workflow_executor

    def _create_intelligent_executor(self, user_id: str):
        """Create executor that routes based on intent analysis (enhanced_builder pattern)."""

        async def intelligent_executor(state: WorkflowState) -> WorkflowState:
            """Route and execute based on intent classification with enhanced routing."""
            try:
                intent_analysis = getattr(state, "intent_classification", None)

                if not intent_analysis:
                    return await self._execute_chat_flow(state, user_id)

                primary_intent = getattr(intent_analysis, "primary_intent", "").lower()
                complexity = getattr(intent_analysis, "complexity_level", None)

                # Intelligent routing based on intent with enhanced logic
                if "research" in primary_intent or "analysis" in primary_intent:
                    composer_logger.logger.info(
                        "Routing to research flow",
                        extra={"user_id": user_id, "intent": primary_intent},
                    )
                    return await self._execute_research_flow(state, user_id)
                elif "creative" in primary_intent or "generate" in primary_intent:
                    composer_logger.logger.info(
                        "Routing to creative flow",
                        extra={"user_id": user_id, "intent": primary_intent},
                    )
                    return await self._execute_creative_flow(state, user_id)
                elif complexity and str(complexity).upper() in [
                    "COMPLEX",
                    "SPECIALIZED",
                ]:
                    composer_logger.logger.info(
                        "Routing to multi-agent flow",
                        extra={"user_id": user_id, "complexity": complexity},
                    )
                    return await self._execute_multi_agent_flow(state, user_id)
                else:
                    composer_logger.logger.info(
                        "Routing to chat flow",
                        extra={"user_id": user_id, "intent": primary_intent},
                    )
                    return await self._execute_chat_flow(state, user_id)

            except Exception as e:
                composer_logger.logger.error(
                    "Intelligent executor failed",
                    extra={"user_id": user_id, "error": str(e)},
                )
                return await self._execute_chat_flow(state, user_id)

        return intelligent_executor

    async def _execute_chat_flow(
        self, state: WorkflowState, user_id: str
    ) -> WorkflowState:
        """Execute optimized chat flow with enhanced execution pattern."""
        try:
            # Simple chat response with streaming
            chat_agent = PipelineNode(
                self.pipeline_factory, ModelProfileType.Primary, stream=True
            )
            state = await chat_agent(state)

            # Handle tools if needed (integration with existing tool system)
            if (
                state.messages
                and hasattr(state.messages[-1], "tool_calls")
                and getattr(state.messages[-1], "tool_calls", None)
                and getattr(state, "required_tools", None)
            ):
                # Tool execution integration with existing ToolRegistry system
                composer_logger.logger.info(
                    "Tool execution requested in chat flow",
                    extra={"user_id": user_id, "tool_count": len(state.required_tools)},
                )

            return state

        except Exception as e:
            composer_logger.logger.error(
                "Chat flow execution failed",
                extra={"user_id": user_id, "error": str(e)},
            )
            # Return minimal response on error - continue with existing state
            return state

    async def _execute_research_flow(
        self, state: WorkflowState, user_id: str
    ) -> WorkflowState:
        """Execute research-focused flow with enhanced execution pattern."""
        try:
            # Query expansion for better research
            query_expander = PipelineNode(
                self.pipeline_factory, ModelProfileType.Analysis, stream=False
            )
            state = await query_expander(state)

            # Enhanced RAG for research using existing infrastructure
            enhanced_rag = EnhancedRAGExecutor(user_id)
            state = await enhanced_rag(state)

            # Research synthesis
            synthesizer = PipelineNode(
                self.pipeline_factory, ModelProfileType.Primary, stream=True
            )
            state = await synthesizer(state)

            composer_logger.logger.info(
                "Research flow completed", extra={"user_id": user_id}
            )

            return state

        except Exception as e:
            composer_logger.logger.error(
                "Research flow execution failed",
                extra={"user_id": user_id, "error": str(e)},
            )
            # Fallback to chat flow
            return await self._execute_chat_flow(state, user_id)

    async def _execute_creative_flow(
        self, state: WorkflowState, user_id: str
    ) -> WorkflowState:
        """Execute creative generation flow with enhanced execution pattern."""
        try:
            # Creative planning
            planner = PipelineNode(
                self.pipeline_factory, ModelProfileType.Primary, stream=False
            )
            state = await planner(state)

            # Content generation
            generator = PipelineNode(
                self.pipeline_factory, ModelProfileType.Primary, stream=True
            )
            state = await generator(state)

            # Refinement (if SelfCritique profile exists)
            if hasattr(ModelProfileType, "SelfCritique"):
                refiner = PipelineNode(
                    self.pipeline_factory, ModelProfileType.SelfCritique, stream=False
                )
                state = await refiner(state)

            composer_logger.logger.info(
                "Creative flow completed", extra={"user_id": user_id}
            )

            return state

        except Exception as e:
            composer_logger.logger.error(
                "Creative flow execution failed",
                extra={"user_id": user_id, "error": str(e)},
            )
            # Fallback to chat flow
            return await self._execute_chat_flow(state, user_id)

    async def _execute_multi_agent_flow(
        self, state: WorkflowState, user_id: str
    ) -> WorkflowState:
        """Execute multi-agent coordination flow with enhanced execution pattern."""
        try:
            # Specialist agent coordination using existing infrastructure
            specialist1 = EngineeringAgentNode(self.pipeline_factory)
            state = await specialist1(state)

            # Coordination agent
            coordinator = PipelineNode(
                self.pipeline_factory, ModelProfileType.Primary, stream=True
            )
            state = await coordinator(state)

            # Final synthesis
            synthesizer = PipelineNode(
                self.pipeline_factory, ModelProfileType.Primary, stream=True
            )
            state = await synthesizer(state)

            composer_logger.logger.info(
                "Multi-agent flow completed", extra={"user_id": user_id}
            )

            return state

        except Exception as e:
            composer_logger.logger.error(
                "Multi-agent flow execution failed",
                extra={"user_id": user_id, "error": str(e)},
            )
            # Fallback to chat flow
            return await self._execute_chat_flow(state, user_id)
