"""
GraphBuilder - Clean facade over simplified workflow architecture.
Maintains API compatibility while using much cleaner internal implementation.
"""

from typing import Any, Dict
from langgraph.graph.state import CompiledStateGraph
from models import WorkflowType
from runner import PipelineFactory
from composer.monitoring.logging import composer_logger
from .simple_builder import GraphBuilder as SimpleGraphBuilder


class GraphBuilder:
    """
    Clean GraphBuilder facade - delegates to simplified implementation.
    
    Maintains API compatibility while using much cleaner internal architecture.
    All the complex logic has been broken down into focused, single-responsibility classes.
    """

    def __init__(self, pipeline_factory: PipelineFactory):
        self._simple_builder = SimpleGraphBuilder(pipeline_factory)
        
        composer_logger.logger.info(
            "GraphBuilder initialized with clean architecture",
            extra={
                "has_pipeline_factory": pipeline_factory is not None,
                "architecture": "simplified_composition",
            },
        )

    async def build_from_context(
        self, user_id: str, workflow_type: WorkflowType
    ) -> CompiledStateGraph:
        """
        Build workflow from user configuration, tools, and workflow type.
        
        Delegates to simplified builder implementation.
        """
        return await self._simple_builder.build_from_context(user_id, workflow_type)

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

            # Add enhanced tool orchestration (replaces simple tool collection)
            tool_collection_fn = await self.create_enhanced_tool_orchestration(user_id)
            workflow.add_node("tool_orchestration", tool_collection_fn)

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
                workflow.add_edge("intent_analysis", "tool_orchestration")
                workflow.add_edge("tool_orchestration", "enhanced_executor")
                workflow.add_edge("enhanced_executor", END)

            else:
                # Complex subgraph routing (preserve existing comprehensive capabilities)
                # Add router node with dedicated WorkflowRouter
                workflow.add_node("router", WorkflowRouter(user_id))

                # Create and add subgraphs as compiled nodes
                subgraphs = await self._create_all_subgraphs(user_id)

                # Add subgraph nodes
                for name, subgraph in subgraphs.items():
                    workflow.add_node(f"{name}_subgraph", subgraph)

                # Add execution coordinator node
                workflow.add_node("coordinator", self._create_coordinator_node(user_id))

                # Define complex workflow edges
                workflow.set_entry_point("intent_analysis")
                workflow.add_edge("intent_analysis", "tool_orchestration")
                workflow.add_edge("tool_orchestration", "router")

                # Conditional routing from router to subgraphs using WorkflowRouter
                router_instance = WorkflowRouter(user_id)
                workflow.add_conditional_edges(
                    "router",
                    router_instance.get_routing_target,
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
        Build a research workflow using dedicated workflow implementation with caching.

        This method delegates to the canonical research workflow definition
        to eliminate code duplication and uses WorkflowCache for performance.
        """
        try:
            # Generate cache key for this workflow configuration
            cache_key = f"research_{user_id}"

            # Check cache first
            if cached := await self._workflow_cache.get(cache_key):
                return cached

            # Use dedicated workflow implementation - tools collected during execution
            if not self.pipeline_factory:
                raise WorkflowConstructionError("Pipeline factory not initialized")

            compiled_workflow = await build_research_workflow(
                user_id=user_id, pipeline_factory=self.pipeline_factory
            )

            # Cache the result
            await self._workflow_cache.set(cache_key, compiled_workflow)
            return compiled_workflow

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
        Build multi-agent orchestration workflow using dedicated workflow implementation with caching.

        This method delegates to the canonical multi-agent workflow definition
        to eliminate code duplication and uses WorkflowCache for performance.
        """
        try:
            # Check cache first
            cache_key = f"multi_agent_{user_id}"
            if cached := await self._workflow_cache.get(cache_key):
                return cached

            # Use dedicated workflow implementation - tools collected during execution
            if not self.pipeline_factory:
                raise WorkflowConstructionError("Pipeline factory not initialized")

            compiled_workflow = await build_multi_agent_workflow(
                user_id=user_id, pipeline_factory=self.pipeline_factory
            )

            # Cache the result
            await self._workflow_cache.set(cache_key, compiled_workflow)

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
        Build creative content generation workflow using dedicated workflow implementation with caching.

        This method delegates to the canonical creative workflow definition
        to eliminate code duplication and uses WorkflowCache for performance.
        """
        try:
            # Check cache first
            cache_key = f"creative_{user_id}"
            if cached := await self._workflow_cache.get(cache_key):
                return cached

            # Use dedicated workflow implementation - tools collected during execution
            if not self.pipeline_factory:
                raise WorkflowConstructionError("Pipeline factory not initialized")

            compiled_workflow = await build_creative_workflow(
                user_id=user_id, pipeline_factory=self.pipeline_factory
            )

            # Cache the result
            await self._workflow_cache.set(cache_key, compiled_workflow)

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
                "chat": await self._build_chat_subgraph(user_id),
                "research": await self.build_research_workflow(user_id),
                "creative": await self.build_creative_workflow(user_id),
                "multi_agent": await self.build_multi_agent_workflow(user_id),
            }
            return subgraphs
        except Exception as e:
            composer_logger.logger.error(f"Failed to create subgraphs: {e}")
            # Return minimal chat subgraph as fallback
            return {"chat": await self._build_chat_subgraph(user_id)}

    async def _build_chat_subgraph(self, user_id: str) -> CompiledStateGraph:
        """Build chat workflow using dedicated workflow implementation with caching."""
        try:
            # Generate cache key for chat subgraph
            cache_key = "chat_subgraph"

            # Check cache first
            if cached := await self._workflow_cache.get(cache_key):
                return cached

            # Use dedicated workflow implementation - tools collected during execution
            if not self.pipeline_factory:
                raise WorkflowConstructionError("Pipeline factory not initialized")

            compiled_workflow = await build_chat_workflow(
                user_id=user_id,
                pipeline_factory=self.pipeline_factory,
            )

            # Cache the result
            await self._workflow_cache.set(cache_key, compiled_workflow)
            return compiled_workflow

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

    # Routing methods removed - now handled by dedicated WorkflowRouter class

    async def _create_fallback_workflow(
        self, user_id: str
    ) -> CompiledStateGraph:  # noqa: ARG002
        """Create minimal fallback workflow."""
        try:
            return await self._build_chat_subgraph(user_id)
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

    # Router node creation removed - now handled by dedicated WorkflowRouter class

    async def create_enhanced_tool_orchestration(self, user_id: str):
        """
        Create enhanced tool orchestration subgraph integration.

        Provides sophisticated tool generation and management capabilities:
        - Static tool retrieval from ToolRegistry
        - Dynamic tool specification generation using engineering models
        - Tool compilation and validation
        - Deduplication and optimization
        - Metadata generation for workflow context
        """
        try:
            composer_logger.logger.info(
                "Creating enhanced tool orchestration subgraph",
                extra={"user_id": user_id},
            )

            # Create tool orchestration subgraph
            tool_subgraph = await create_tool_orchestration_subgraph(
                pipeline_factory=self.pipeline_factory
            )

            async def enhanced_tool_orchestration(
                state: WorkflowState,
            ) -> WorkflowState:
                """Execute enhanced tool orchestration using dedicated subgraph."""
                try:
                    # Create tool orchestration state from workflow state
                    # Extract user query from messages
                    user_query = ""
                    if state.messages:
                        for msg in reversed(
                            state.messages
                        ):  # Get the latest user message
                            if getattr(msg, "type", None) == "human":
                                user_query = getattr(msg, "content", "")
                                break

                    tool_state = ToolOrchestrationState(
                        user_id=state.user_id or "unknown",
                        user_query=user_query,
                        intent_analysis=state.intent_classification
                        or IntentAnalysis(
                            primary_intent="general_assistance",
                            complexity_level=ComplexityLevel.MODERATE,
                            required_capabilities=[
                                RequiredCapability.INFORMATION_RETRIEVAL,
                                RequiredCapability.TEXT_PROCESSING,
                            ],
                            computational_requirements=[
                                ComputationalRequirement.EXTERNAL_API_CALLS
                            ],
                            domain_specificity=0.5,
                            reusability_potential=0.7,
                            confidence=0.8,
                        ),
                    )

                    # Execute tool orchestration subgraph
                    result_state = tool_subgraph.invoke(tool_state)

                    # Merge results back to workflow state
                    state.required_tools = getattr(result_state, "required_tools", [])

                    # Update metadata
                    if state.execution_metadata.tool_orchestration is None:
                        state.execution_metadata.tool_orchestration = {}

                    state.execution_metadata.tool_orchestration.update(
                        {
                            "dynamic_tools_generated": len(
                                [
                                    t
                                    for t in state.required_tools
                                    if getattr(t, "is_dynamic", False)
                                ]
                            ),
                            "static_tools_selected": len(
                                [
                                    t
                                    for t in state.required_tools
                                    if not getattr(t, "is_dynamic", False)
                                ]
                            ),
                            "orchestration_completed": True,
                        }
                    )

                    composer_logger.logger.info(
                        "Enhanced tool orchestration completed",
                        extra={
                            "user_id": user_id,
                            "tool_count": len(state.required_tools),
                            "dynamic_tools": state.execution_metadata.tool_orchestration[
                                "dynamic_tools_generated"
                            ],
                            "static_tools": state.execution_metadata.tool_orchestration[
                                "static_tools_selected"
                            ],
                        },
                    )
                    return state

                except Exception as e:
                    composer_logger.logger.error(
                        "Enhanced tool orchestration failed, falling back to basic collection",
                        extra={"user_id": user_id, "error": str(e)},
                    )
                    # Fallback to basic tool collection
                    tool_registry = ToolRegistry()
                    intent = getattr(state, "intent_classification", None)
                    if intent:
                        tools = await tool_registry.get_tools_for_context(
                            intent, user_id
                        )
                        state.required_tools = tools or []
                    else:
                        state.required_tools = []
                    return state

            return enhanced_tool_orchestration

        except Exception as e:
            composer_logger.logger.error(
                "Failed to create enhanced tool orchestration, falling back to basic collection",
                extra={"user_id": user_id, "error": str(e)},
            )
            # Fallback to basic tool collection
            tool_registry = ToolRegistry()

            async def fallback_tool_collection(state: WorkflowState) -> WorkflowState:
                try:
                    intent = getattr(state, "intent_classification", None)
                    if intent:
                        tools = await tool_registry.get_tools_for_context(
                            intent, user_id
                        )
                        state.required_tools = tools or []
                    else:
                        state.required_tools = []
                    return state
                except Exception as fallback_error:
                    composer_logger.logger.error(
                        f"Fallback tool collection failed: {fallback_error}"
                    )
                    state.required_tools = []
                    return state

            return fallback_tool_collection

    async def integrate_memory_and_context_assembly(
        self, workflow_graph: StateGraph, user_id: str
    ) -> StateGraph:
        """
        Integrate memory operations and context assembly into existing workflow.

        Uses existing memory workflow and unified context assembler to provide
        context extension capabilities without redundant workflow layers.

        Args:
            workflow_graph: Existing workflow graph to enhance
            user_id: User identifier for configuration

        Returns:
            Enhanced workflow graph with memory and context assembly capabilities
        """
        try:
            from composer.nodes.memory import MemorySearchNode
            from composer.agents.unified_context_assembler import (
                UnifiedContextAssembler,
            )

            composer_logger.logger.info(
                "Integrating memory operations and context assembly",
                extra={
                    "user_id": user_id,
                    "operation": "integrate_memory_context_assembly",
                },
            )

            # Create memory search node for memory operations
            memory_node = MemorySearchNode()

            # Create context assembly function using unified context assembler
            context_assembler = UnifiedContextAssembler()

            async def assemble_context_node(state):
                """Node function for context assembly using UnifiedContextAssembler."""
                assembled_messages = context_assembler.assemble_context(state)
                return {"messages": assembled_messages}

            # Add nodes to workflow
            workflow_graph.add_node("memory_operations", memory_node)
            workflow_graph.add_node("assemble_context", assemble_context_node)

            composer_logger.logger.info(
                "Memory and context assembly integration completed",
                extra={
                    "user_id": user_id,
                    "nodes_added": ["memory_operations", "assemble_context"],
                },
            )

            return workflow_graph

        except Exception as e:
            composer_logger.logger.error(
                "Memory and context assembly integration failed",
                extra={
                    "user_id": user_id,
                    "error": str(e),
                    "operation": "integrate_memory_context_assembly",
                },
            )
            raise WorkflowConstructionError(
                f"Memory and context assembly integration failed: {e}"
            ) from e

    async def create_memory_enhanced_workflow(self, user_id: str) -> CompiledStateGraph:
        """
        Create workflow enhanced with memory operations using existing workflows.

        Composes memory workflow with context assembly to provide context extension
        without redundant workflow layers.

        Args:
            user_id: User identifier for configuration

        Returns:
            Memory-enhanced workflow using existing components
        """
        try:
            composer_logger.logger.info(
                "Creating memory-enhanced workflow",
                extra={
                    "user_id": user_id,
                    "operation": "create_memory_enhanced_workflow",
                },
            )

            # Use existing memory workflow as foundation
            memory_workflow = await build_memory_workflow(
                user_id=user_id,
                pipeline_factory=self.pipeline_factory,
                store_memories=True,
            )

            composer_logger.logger.info(
                "Memory-enhanced workflow created successfully",
                extra={"user_id": user_id, "base": "memory_workflow"},
            )

            return memory_workflow

        except Exception as e:
            composer_logger.logger.error(
                "Memory-enhanced workflow creation failed",
                extra={"user_id": user_id, "error": str(e)},
            )
            raise WorkflowConstructionError(
                f"Memory-enhanced workflow creation failed: {e}"
            ) from e

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

            # Deep search for research using modern search architecture
            from composer.nodes.research import (  # pylint: disable=import-outside-toplevel
                ComprehensiveResearchExecutor,
            )

            deep_search = ComprehensiveResearchExecutor(user_id)
            state = await deep_search(state)

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

    # Cache Management Methods

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get workflow cache statistics for monitoring and debugging."""
        return await self._workflow_cache.get_stats()

    async def clear_cache(self) -> None:
        """Clear all cached workflows (useful for debugging or config changes)."""
        await self._workflow_cache.clear()
        composer_logger.logger.info("GraphBuilder workflow cache cleared")

    async def invalidate_user_workflows(self, user_id: str) -> None:
        """Invalidate cached workflows for a specific user (e.g., when user config changes)."""
        # For now, we clear all cache since we don't have user-specific keys stored
        # This could be optimized in the future to track keys by user_id
        await self._workflow_cache.clear()
        composer_logger.logger.info(
            f"All workflows invalidated due to user config change for {user_id}"
        )

    async def close(self) -> None:
        """Clean up GraphBuilder resources including workflow cache."""
        await self._workflow_cache.close()
        composer_logger.logger.info("GraphBuilder closed and cache cleaned up")
