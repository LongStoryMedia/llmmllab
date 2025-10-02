"""
GraphBuilder for dynamic workflow construction.
Constructs LangGraph workflows dynamically based on conversation context and tools.
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

    async def build_from_context(
        self,
        user_id: str,
        messages: List[Message],
        tools: List[AvailableTool],
        workflow_type: str,
    ) -> Any:
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
                    "tool_count": len(tools),
                    "user_id": user_id,
                },
            )

            # Select appropriate build method based on workflow type
            if workflow_type == WorkflowType.CHAT:
                return await self.build_chat_workflow(user_id, messages, tools)
            elif workflow_type == WorkflowType.RESEARCH:
                return await self.build_research_workflow(user_id, messages, tools)
            elif workflow_type == WorkflowType.MULTI_AGENT:
                return await self.build_multi_agent_workflow(user_id, messages, tools)
            elif workflow_type == WorkflowType.CREATIVE:
                return await self.build_creative_workflow(user_id, messages, tools)
            else:
                # Default to chat workflow
                return await self.build_chat_workflow(user_id, messages, tools)

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
        Build master workflow with intelligent routing and optional explicit workflow type.
        
        Args:
            user_id: User ID for configuration retrieval
            workflow_type: Optional explicit workflow type. If None, uses intent analysis.
        
        Returns:
            CompiledStateGraph: Master workflow with intelligent routing
        """
        try:
            composer_logger.logger.info(
                "Building master workflow",
                extra={"user_id": user_id, "workflow_type": workflow_type}
            )
            
            if workflow_type:
                # For explicit workflow type, route directly to that workflow
                if workflow_type == WorkflowType.RESEARCH:
                    return await self.build_research_workflow(user_id, [], [])
                elif workflow_type == WorkflowType.CREATIVE:
                    return await self.build_creative_workflow(user_id, [], [])
                elif workflow_type == WorkflowType.MULTI_AGENT:
                    return await self.build_multi_agent_workflow(user_id, [], [])
                else:  # Default to chat
                    return await self.build_from_context(user_id, [], [], WorkflowType.CHAT)
            else:
                # For intelligent routing, analyze context and build appropriate workflow
                # TODO: Implement proper intent analysis here
                # For now, default to chat workflow
                return await self.build_from_context(user_id, [], [], WorkflowType.CHAT)
                
        except Exception as e:
            composer_logger.logger.error(
                "Failed to build master workflow",
                extra={"user_id": user_id, "error": str(e)}
            )
            # Fallback to chat workflow
            return await self.build_from_context(user_id, [], [], WorkflowType.CHAT)

    async def build_research_workflow(
        self, user_id: str, messages: List[Message], tools: List[AvailableTool]
    ) -> Any:
        """
        Build a research workflow with deep RAG and synthesis capabilities.

        This workflow emphasizes comprehensive information gathering,
        multi-source analysis, and detailed synthesis.
        """
        try:
            composer_logger.logger.info(
                "Building research workflow",
                extra={"user_id": user_id, "tool_count": len(tools)},
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

    async def build_multi_agent_workflow(
        self,
        user_id: str,
        messages: List[Message],
        tools: List[AvailableTool],  # noqa: ARG002
    ) -> Any:
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
                extra={"user_id": user_id, "tool_count": len(tools)},
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

    async def build_creative_workflow(
        self,
        user_id: str,
        messages: List[Message],
        tools: List[AvailableTool],  # noqa: ARG002
    ) -> CompiledStateGraph:
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
                extra={"user_id": user_id, "tool_count": len(tools)},
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
