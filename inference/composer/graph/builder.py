"""
GraphBuilder for dynamic workflow construction.
Constructs LangGraph workflows dynamically based on conversation context and tools.
"""
from typing import Dict, Any, List, Optional
import sys
sys.path.append('/Users/lons7862/workspace/llmmllab/inference')

from models.conversation_ctx import ConversationCtx
from models.available_tool import AvailableTool
from composer.monitoring.logging import composer_logger
from composer.core.errors import WorkflowConstructionError


class CompiledGraph:
    """
    Placeholder for LangGraph compiled graph.
    In production, this would be a proper LangGraph StateGraph.compile() result.
    """
    
    def __init__(self, workflow_type: str, nodes: List[str], config: Dict[str, Any]):
        self.workflow_type = workflow_type
        self.nodes = nodes
        self.config = config
    
    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the workflow with given state."""
        # Placeholder implementation
        composer_logger.logger.info(
            "Executing workflow", 
            extra={
                "workflow_type": self.workflow_type,
                "node_count": len(self.nodes)
            }
        )
        return state
    
    async def astream_events(self, state: Dict[str, Any], version: str = "v2"):
        """Stream workflow execution events."""
        # Placeholder implementation for streaming
        yield {"event": "workflow_start", "data": {"workflow_type": self.workflow_type}}
        
        for node in self.nodes:
            yield {"event": "node_start", "data": {"node": node}}
            yield {"event": "node_end", "data": {"node": node}}
        
        yield {"event": "workflow_end", "data": state}


class GraphBuilder:
    """
    Constructs LangGraph workflows dynamically based on context.
    
    The GraphBuilder implements the core workflow construction logic,
    supporting different workflow types with appropriate node compositions.
    """
    
    def __init__(self):
        composer_logger.logger.info("GraphBuilder initialized")
    
    async def build_from_context(
        self,
        conversation_ctx: ConversationCtx,
        tools: List[AvailableTool],
        config: Dict[str, Any],
        workflow_type: str
    ) -> CompiledGraph:
        """
        Build workflow from context, tools, and configuration.
        
        This is the main entry point for dynamic workflow construction.
        """
        try:
            composer_logger.logger.info(
                "Building workflow from context",
                extra={
                    "workflow_type": workflow_type,
                    "tool_count": len(tools),
                    "config_keys": list(config.keys())
                }
            )
            
            # Select appropriate build method based on workflow type
            if workflow_type == "CHAT":
                return await self.build_chat_workflow(conversation_ctx, tools, config)
            elif workflow_type == "RESEARCH":
                return await self.build_research_workflow(conversation_ctx, tools, config)
            elif workflow_type == "MULTI_AGENT":
                return await self.build_multi_agent_workflow(conversation_ctx, tools, config)
            elif workflow_type == "CREATIVE":
                return await self.build_creative_workflow(conversation_ctx, tools, config)
            else:
                # Default to chat workflow
                return await self.build_chat_workflow(conversation_ctx, tools, config)
            
        except Exception as e:
            composer_logger.log_error(e, {"context": "workflow_construction", "workflow_type": workflow_type})
            raise WorkflowConstructionError(f"Failed to build {workflow_type} workflow: {e}")
    
    async def build_chat_workflow(
        self,
        conversation_ctx: ConversationCtx,
        tools: List[AvailableTool],
        config: Dict[str, Any]
    ) -> CompiledGraph:
        """
        Build standard chat workflow with RAG and tool support.
        
        Workflow: RAG Enrichment -> Dynamic Tools -> Agent -> Tool Execution (conditional)
        """
        nodes = []
        
        # Always include RAG enrichment for context
        nodes.append("rag_enrichment")
        
        # Add dynamic tools if enabled
        if config.get("enable_tool_generation", False):
            nodes.append("dynamic_tools")
        
        # Primary chat agent (with streaming enabled)
        nodes.append("agent")
        
        # Tool execution if tools are available
        if tools:
            nodes.append("tools")
        
        workflow_config = {
            **config,
            "streaming_enabled": config.get("streaming_enabled", True),
            "tools": [tool.dict() for tool in tools]
        }
        
        compiled_graph = CompiledGraph("CHAT", nodes, workflow_config)
        
        composer_logger.logger.info(
            "Built chat workflow",
            extra={
                "node_count": len(nodes),
                "nodes": nodes,
                "streaming_enabled": workflow_config["streaming_enabled"]
            }
        )
        
        return compiled_graph
    
    async def build_research_workflow(
        self,
        conversation_ctx: ConversationCtx,
        tools: List[AvailableTool],
        config: Dict[str, Any]
    ) -> CompiledGraph:
        """
        Build research workflow with configurable RAG depth.
        
        Workflow: Intent Classification -> Conditional RAG (Shallow/Deep) -> Synthesis -> Response
        """
        nodes = [
            "intent_classification",
            "rag_router",  # Conditional node for RAG depth
            "parallel_search",
            "synthesis",
            "response_generation"
        ]
        
        workflow_config = {
            **config,
            "rag_depth": config.get("rag_depth", "DEEP"),
            "max_sources": config.get("max_sources", 10),
            "retrieve_full_content": config.get("retrieve_full_content", True),
            "tools": [tool.dict() for tool in tools]
        }
        
        compiled_graph = CompiledGraph("RESEARCH", nodes, workflow_config)
        
        composer_logger.logger.info(
            "Built research workflow",
            extra={
                "node_count": len(nodes),
                "rag_depth": workflow_config["rag_depth"],
                "max_sources": workflow_config["max_sources"]
            }
        )
        
        return compiled_graph
    
    async def build_multi_agent_workflow(
        self,
        conversation_ctx: ConversationCtx,
        tools: List[AvailableTool],
        config: Dict[str, Any]
    ) -> CompiledGraph:
        """
        Build multi-agent orchestration workflow.
        
        Workflow: Agent Router -> Specialized Agents -> Coordination -> Final Response
        """
        nodes = [
            "agent_router",
            "specialist_agent_1",
            "specialist_agent_2", 
            "coordination",
            "final_response"
        ]
        
        workflow_config = {
            **config,
            "enable_handoffs": True,
            "max_agent_iterations": config.get("max_agent_iterations", 5),
            "tools": [tool.dict() for tool in tools]
        }
        
        compiled_graph = CompiledGraph("MULTI_AGENT", nodes, workflow_config)
        
        composer_logger.logger.info(
            "Built multi-agent workflow",
            extra={
                "node_count": len(nodes),
                "max_iterations": workflow_config["max_agent_iterations"]
            }
        )
        
        return compiled_graph
    
    async def build_creative_workflow(
        self,
        conversation_ctx: ConversationCtx,
        tools: List[AvailableTool],
        config: Dict[str, Any]
    ) -> CompiledGraph:
        """
        Build creative content generation workflow.
        
        Workflow: Creative Planning -> Content Generation -> Refinement -> Output
        """
        nodes = [
            "creative_planning",
            "content_generation",
            "refinement",
            "output_formatting"
        ]
        
        workflow_config = {
            **config,
            "creative_mode": config.get("creative_mode", "balanced"),
            "refinement_iterations": config.get("refinement_iterations", 2),
            "tools": [tool.dict() for tool in tools]
        }
        
        compiled_graph = CompiledGraph("CREATIVE", nodes, workflow_config)
        
        composer_logger.logger.info(
            "Built creative workflow",
            extra={
                "node_count": len(nodes),
                "creative_mode": workflow_config["creative_mode"],
                "refinement_iterations": workflow_config["refinement_iterations"]
            }
        )
        
        return compiled_graph