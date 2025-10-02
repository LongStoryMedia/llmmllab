"""
GraphBuilder for dynamic workflow construction.
Constructs LangGraph workflows dynamically based on conversation context and tools.
"""

from typing import Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from models.workflow_type import WorkflowType
from models import ModelProfileType
from composer.monitoring.logging import composer_logger
from composer.core.errors import WorkflowConstructionError
from composer.graph.state import WorkflowState, ResearchWorkflowState

# Node imports
from composer.nodes.standard import PipelineNode
from composer.nodes.intent_classifier import IntentClassifierNode
from composer.nodes.engineering_agent import EngineeringAgentNode
from composer.nodes.rag.executor import EnhancedRAGExecutor
from composer.tools.registry import ToolRegistry

# Database import - lazy loaded to avoid circular dependencies
from db import storage  # pylint: disable=import-outside-toplevel


class GraphBuilder:
    """
    Constructs LangGraph workflows dynamically based on context.

    The GraphBuilder implements the core workflow construction logic,
    supporting different workflow types with appropriate node compositions.
    """

    def __init__(self, pipeline_factory=None):
        self.pipeline_factory = pipeline_factory
        composer_logger.logger.info(
            "GraphBuilder initialized",
            has_pipeline_factory=pipeline_factory is not None,
        )

    async def _get_user_config(self, user_id: str):
        """Get user configuration from shared data layer."""
        try:
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

    async def build_from_context(self, user_id: str, workflow_type: str) -> Any:
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
                extra={"user_id": user_id, "workflow_type": workflow_type}
            )
            
            # Create master workflow graph
            workflow = StateGraph(WorkflowState)
            
            # Add intent analysis node (always present for context enrichment)
            workflow.add_node("intent_analysis", IntentClassifierNode())
            
            # Add tool collection node (collects available tools based on intent)
            workflow.add_node("tool_collection", self._create_tool_collection_node(user_id))
            
            # Add router node with routing logic
            workflow.add_node("router", self._create_router_node(user_id, workflow_type))
            
            # Create and add subgraphs as compiled nodes
            subgraphs = await self._create_all_subgraphs(user_id)
            
            # Add subgraph nodes
            for name, subgraph in subgraphs.items():
                workflow.add_node(f"{name}_subgraph", subgraph)
            
            # Add execution coordinator node
            workflow.add_node("coordinator", self._create_coordinator_node(user_id))
            
            # Define workflow edges
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
                    "coordinator": "coordinator"  # For parallel/series execution
                }
            )
            
            # All subgraphs route to coordinator for result processing
            for name in subgraphs.keys():
                workflow.add_edge(f"{name}_subgraph", "coordinator")
            
            workflow.add_edge("coordinator", END)
            
            # Compile and return the master workflow
            compiled_workflow = workflow.compile()
            
            composer_logger.logger.info(
                "Master workflow compiled successfully",
                extra={"user_id": user_id, "subgraph_count": len(subgraphs)}
            )
            
            return compiled_workflow
                
        except Exception as e:
            composer_logger.logger.error(
                "Failed to build master workflow",
                extra={"user_id": user_id, "error": str(e)}
            )
            # Create simple fallback workflow
            return await self._create_fallback_workflow(user_id)

    async def build_research_workflow(self, user_id: str) -> Any:
        """
        Build a research workflow with deep RAG and synthesis capabilities.

        This workflow emphasizes comprehensive information gathering,
        multi-source analysis, and detailed synthesis.
        """
        try:
            composer_logger.logger.info(
                "Building research workflow",
                extra={"user_id": user_id},
            )

            # Create research workflow graph
            workflow = StateGraph(ResearchWorkflowState)

            # Configuration retrieved internally from shared data layer using user_id
            user_config = await self._get_user_config(user_id)
            if not user_config:
                raise WorkflowConstructionError("Unable to retrieve user configuration")

            # Use injected pipeline factory
            pipeline_factory = self.pipeline_factory

            # Add research-specific nodes
            workflow.add_node("intent_classifier", IntentClassifierNode())
            workflow.add_node(
                "query_expansion",
                PipelineNode(
                    pipeline_factory,
                    ModelProfileType.Analysis,
                    stream=False,  # Analysis doesn't need streaming
                ),
            )
            workflow.add_node("enhanced_rag", EnhancedRAGExecutor(user_id))
            workflow.add_node(
                "synthesis_agent",
                PipelineNode(
                    pipeline_factory,
                    ModelProfileType.Primary,
                    stream=True,  # Final synthesis can stream
                ),
            )

            # Set up research workflow flow
            workflow.set_entry_point("intent_classifier")
            workflow.add_edge("intent_classifier", "query_expansion")
            workflow.add_edge("query_expansion", "enhanced_rag")
            workflow.add_edge("enhanced_rag", "synthesis_agent")
            workflow.add_edge("synthesis_agent", END)

            # Compile and return workflow
            compiled_workflow = workflow.compile()

            composer_logger.logger.info(
                "Research workflow built successfully",
                extra={"user_id": user_id, "nodes": len(workflow.nodes)},
            )

            return compiled_workflow

        except Exception as e:
            composer_logger.logger.error(
                "Failed to build research workflow",
                extra={"user_id": user_id, "error": str(e)},
            )
            raise WorkflowConstructionError(
                f"Research workflow construction failed: {e}"
            ) from e

    async def build_multi_agent_workflow(self, user_id: str) -> Any:
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
                "multi_agent": await self.build_multi_agent_workflow(user_id)
            }
            return subgraphs
        except Exception as e:
            composer_logger.logger.error(f"Failed to create subgraphs: {e}")
            # Return minimal chat subgraph as fallback
            return {"chat": await self._build_chat_subgraph()}

    async def _build_chat_subgraph(self) -> CompiledStateGraph:
        """Build simple chat subgraph."""
        workflow = StateGraph(WorkflowState)
        
        # Simple chat pipeline
        pipeline_factory = self.pipeline_factory
        workflow.add_node(
            "chat_response",
            PipelineNode(
                pipeline_factory,
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
                        "workflows": selected_workflows
                    }
                )
                
                # For now, just pass through - coordination logic can be enhanced
                state.final_response = getattr(state, "response", "Processing complete")
                return state
                
            except Exception as e:
                composer_logger.logger.error(
                    "Coordination failed",
                    extra={"user_id": user_id, "error": str(e)}
                )
                state.final_response = "Sorry, there was an error processing your request."
                return state
                
        return coordinate_execution

    def _route_to_subgraphs(self, state):
        """Determine which subgraph to route to based on state."""
        try:
            execution_strategy = getattr(state, "execution_strategy", "single")
            selected_workflows = getattr(state, "selected_workflows", ["chat"])
            
            # For complex strategies, route to coordinator
            if execution_strategy in ["parallel", "series"] and len(selected_workflows) > 1:
                return "coordinator"
            
            # For single workflow, route directly
            if selected_workflows:
                workflow_type = selected_workflows[0]
                if workflow_type in ["research", "creative", "multi_agent", "chat"]:
                    return workflow_type
            
            # Default fallback
            return "chat"
            
        except Exception as e:
            composer_logger.logger.error(f"Routing failed: {e}")
            return "chat"

    async def _create_fallback_workflow(self, user_id: str) -> CompiledStateGraph:  # noqa: ARG002
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

    def _create_router_node(self, user_id: str, explicit_workflow_type: Optional[WorkflowType] = None):
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
                    state.execution_strategy = "series"  # Research first, then creative synthesis
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
                        "execution_strategy": state.execution_strategy
                    }
                )
                
                return state
                
            except Exception as e:
                composer_logger.logger.error(
                    "Router failed, falling back to chat",
                    extra={"user_id": user_id, "error": str(e)}
                )
                # Fallback to chat workflow
                state.selected_workflows = ["chat"]
                state.execution_strategy = "single"
                return state
                
        return route_workflows

    def _create_tool_collection_node(self, user_id: str):
        """
        Create tool collection node that gathers available tools based on intent analysis.
        
        This replaces the tool collection logic that was previously in the core service.
        Uses ToolRegistry to collect both static and dynamic tools based on intent.
        """
        async def collect_tools(state):
            """Collect available tools based on intent analysis results."""
            try:
                # Get intent from previous node
                intent = getattr(state, 'intent_analysis', None) or getattr(state, 'intent', None)
                
                if not intent:
                    composer_logger.logger.warning(
                        "No intent analysis found, using minimal tool set",
                        extra={"user_id": user_id}
                    )
                    # Set empty tool list and continue\n                    state.required_tools = []
                    return state
                
                composer_logger.logger.info(
                    "Collecting tools for intent",
                    extra={
                        "user_id": user_id,
                        "primary_intent": getattr(intent, 'primary_intent', 'unknown'),
                        "requires_tools": getattr(intent, 'requires_tools', False)
                    }
                )
                
                # Initialize tool registry
                tool_registry = ToolRegistry()
                
                # Get tools for the current intent and user context
                tools = await tool_registry.get_tools_for_context(intent, user_id)
                
                # Store tools in state for subgraphs to use
                # Store tools in state for subgraphs to use\n                state.required_tools = tools
                
                composer_logger.logger.info(
                    "Tool collection completed",
                    extra={
                        "user_id": user_id,
                        "tool_count": len(tools),
                        "tool_names": [tool.name for tool in tools] if tools else []
                    }
                )
                
                return state
                
            except Exception as e:
                composer_logger.logger.error(
                    "Tool collection failed",
                    extra={"user_id": user_id, "error": str(e)}
                )
                # Continue with empty tool list on error\n                state.required_tools = []
                return state
                
        return collect_tools
