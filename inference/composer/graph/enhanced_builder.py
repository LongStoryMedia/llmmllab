"""
Enhanced GraphBuilder with proper master workflow and subgraph implementation.
This implements the proper LangGraph subgraph pattern with explicit workflow type support.
"""

from typing import Any, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from models.available_tool import AvailableTool
from models.workflow_type import WorkflowType
from models import Message, ModelProfileType
from composer.monitoring.logging import composer_logger
from composer.core.errors import WorkflowConstructionError
from composer.graph.state import WorkflowState, ResearchWorkflowState

# Node imports
from composer.nodes.standard import PipelineNode, ToolExecutorNode
from composer.nodes.intent_classifier import IntentClassifierNode
from composer.nodes.engineering_agent import EngineeringAgentNode
from composer.nodes.rag.router import RAGRouter, ShallowRAGExecutor, DeepRAGExecutor
from composer.nodes.rag.executor import EnhancedRAGExecutor


class EnhancedGraphBuilder:
    """Enhanced GraphBuilder with proper master workflow and subgraph support."""
    
    def __init__(self, pipeline_factory=None):
        self.pipeline_factory = pipeline_factory
        composer_logger.logger.info(
            "Enhanced GraphBuilder initialized",
            has_pipeline_factory=pipeline_factory is not None,
        )

    async def build_master_workflow(
        self, user_id: str, workflow_type: Optional[WorkflowType] = None
    ) -> CompiledStateGraph:
        """
        Build master workflow with intelligent routing and optional explicit workflow type.
        
        This implements LangGraph best practices using subgraphs for different workflow types.
        
        Args:
            user_id: User ID for configuration retrieval
            workflow_type: Optional explicit workflow type. If None, uses intent analysis routing.
        
        Returns:
            CompiledStateGraph: Master workflow with intelligent routing
        """
        try:
            composer_logger.logger.info(
                "Building master workflow with subgraphs",
                extra={"user_id": user_id, "explicit_workflow_type": workflow_type}
            )
            
            # Create master workflow graph
            workflow = StateGraph(WorkflowState)
            
            if workflow_type:
                # Explicit workflow type - skip intent analysis, go straight to execution
                workflow.add_node("tool_selection", EngineeringAgentNode(self.pipeline_factory))
                workflow.add_node("execute_workflow", self._create_workflow_executor(user_id, workflow_type))
                
                workflow.set_entry_point("tool_selection")
                workflow.add_edge("tool_selection", "execute_workflow")
                workflow.add_edge("execute_workflow", END)
                
            else:
                # Intelligent routing - analyze intent first, then route
                workflow.add_node("intent_analysis", IntentClassifierNode())
                workflow.add_node("tool_selection", EngineeringAgentNode(self.pipeline_factory))
                workflow.add_node("execute_workflow", self._create_intelligent_executor(user_id))
                
                workflow.set_entry_point("intent_analysis")
                workflow.add_edge("intent_analysis", "tool_selection")
                workflow.add_edge("tool_selection", "execute_workflow")
                workflow.add_edge("execute_workflow", END)
            
            # Compile master workflow
            compiled_workflow = workflow.compile()
            
            composer_logger.logger.info(
                "Master workflow built successfully",
                extra={
                    "user_id": user_id, 
                    "node_count": len(workflow.nodes),
                    "routing_mode": "explicit" if workflow_type else "intelligent"
                }
            )
            
            return compiled_workflow
            
        except Exception as e:
            composer_logger.logger.error(
                "Failed to build master workflow",
                extra={"user_id": user_id, "error": str(e)}
            )
            raise WorkflowConstructionError(
                f"Master workflow construction failed: {e}"
            ) from e

    def _create_workflow_executor(self, user_id: str, workflow_type: WorkflowType):
        """Create executor for explicit workflow type."""
        async def workflow_executor(state: WorkflowState) -> WorkflowState:
            """Execute specific workflow type."""
            try:
                if workflow_type == WorkflowType.RESEARCH:
                    return await self._execute_research_flow(state, user_id)
                elif workflow_type == WorkflowType.CREATIVE:
                    return await self._execute_creative_flow(state, user_id)
                elif workflow_type == WorkflowType.MULTI_AGENT:
                    return await self._execute_multi_agent_flow(state, user_id)
                else:  # Default to chat
                    return await self._execute_chat_flow(state, user_id)
                    
            except Exception as e:
                composer_logger.logger.error(f"Workflow executor failed: {e}")
                return await self._execute_chat_flow(state, user_id)  # Fallback
        
        return workflow_executor

    def _create_intelligent_executor(self, user_id: str):
        """Create executor that routes based on intent analysis."""
        async def intelligent_executor(state: WorkflowState) -> WorkflowState:
            """Route and execute based on intent classification."""
            try:
                intent_analysis = getattr(state, "intent_classification", None)
                
                if not intent_analysis:
                    return await self._execute_chat_flow(state, user_id)
                
                primary_intent = getattr(intent_analysis, "primary_intent", "").lower()
                complexity = getattr(intent_analysis, "complexity_level", None)
                
                # Intelligent routing based on intent
                if "research" in primary_intent or "analysis" in primary_intent:
                    composer_logger.logger.info("Routing to research flow", extra={"user_id": user_id})
                    return await self._execute_research_flow(state, user_id)
                elif "creative" in primary_intent or "generate" in primary_intent:
                    composer_logger.logger.info("Routing to creative flow", extra={"user_id": user_id})
                    return await self._execute_creative_flow(state, user_id)
                elif complexity and str(complexity).upper() in ["COMPLEX", "SPECIALIZED"]:
                    composer_logger.logger.info("Routing to multi-agent flow", extra={"user_id": user_id})
                    return await self._execute_multi_agent_flow(state, user_id)
                else:
                    composer_logger.logger.info("Routing to chat flow", extra={"user_id": user_id})
                    return await self._execute_chat_flow(state, user_id)
                    
            except Exception as e:
                composer_logger.logger.error(f"Intelligent executor failed: {e}")
                return await self._execute_chat_flow(state, user_id)
        
        return intelligent_executor

    async def _execute_chat_flow(self, state: WorkflowState, user_id: str) -> WorkflowState:
        """Execute optimized chat flow."""
        # RAG routing based on intent
        router = RAGRouter()
        rag_route = router.route_rag_depth(state)
        
        # Execute appropriate RAG
        if rag_route == "execute_deep_crawl_and_synthesize":
            rag_executor = DeepRAGExecutor(user_id)
        else:
            rag_executor = ShallowRAGExecutor(user_id)
        
        state = await rag_executor(state)
        
        # Chat response with streaming
        chat_agent = PipelineNode(
            self.pipeline_factory,
            ModelProfileType.Primary,
            stream=True
        )
        state = await chat_agent(state)
        
        # Handle tools if needed (simplified for now)
        if (state.messages and 
            hasattr(state.messages[-1], "tool_calls") and 
            state.messages[-1].tool_calls and
            getattr(state, "required_tools", None)):
            # Tool execution would go here
            pass
        
        return state

    async def _execute_research_flow(self, state: WorkflowState, user_id: str) -> WorkflowState:
        """Execute research-focused flow."""
        # Query expansion
        query_expander = PipelineNode(
            self.pipeline_factory,
            ModelProfileType.Analysis,
            stream=False
        )
        state = await query_expander(state)
        
        # Enhanced RAG for research
        enhanced_rag = EnhancedRAGExecutor(user_id)
        state = await enhanced_rag(state)
        
        # Research synthesis
        synthesizer = PipelineNode(
            self.pipeline_factory,
            ModelProfileType.Primary,
            stream=True
        )
        state = await synthesizer(state)
        
        return state

    async def _execute_creative_flow(self, state: WorkflowState, user_id: str) -> WorkflowState:
        """Execute creative generation flow."""
        # Creative planning
        planner = PipelineNode(
            self.pipeline_factory,
            ModelProfileType.Primary,
            stream=False
        )
        state = await planner(state)
        
        # Content generation
        generator = PipelineNode(
            self.pipeline_factory,
            ModelProfileType.Primary,
            stream=True
        )
        state = await generator(state)
        
        # Refinement (optional)
        if hasattr(ModelProfileType, 'SelfCritique'):
            refiner = PipelineNode(
                self.pipeline_factory,
                ModelProfileType.SelfCritique,
                stream=False
            )
            state = await refiner(state)
        
        return state

    async def _execute_multi_agent_flow(self, state: WorkflowState, user_id: str) -> WorkflowState:
        """Execute multi-agent coordination flow."""
        # Specialist agent 1
        specialist1 = EngineeringAgentNode(self.pipeline_factory)
        state = await specialist1(state)
        
        # Coordination agent
        coordinator = PipelineNode(
            self.pipeline_factory,
            ModelProfileType.Primary,
            stream=True
        )
        state = await coordinator(state)
        
        # Final synthesis
        synthesizer = PipelineNode(
            self.pipeline_factory,
            ModelProfileType.Primary,
            stream=True
        )
        state = await synthesizer(state)
        
        return state