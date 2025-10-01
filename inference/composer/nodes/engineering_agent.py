"""
Engineering Agent Node for LangGraph workflow integration.
Provides LangGraph node wrapper for dynamic tool generation and orchestration.
"""

from models.available_tool import AvailableTool
from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError


class EngineeringAgentNode:
    """
    LangGraph node wrapper for Engineering Agent.
    
    Handles workflow state management and delegates business logic to EngineeringAgent.
    Follows the architectural pattern where nodes wrap agents for LangGraph integration.
    """

    def __init__(self, pipeline_factory):
        """
        Initialize engineering agent node.
        
        Args:
            pipeline_factory: Factory for creating structured pipelines
        """
        from composer.agents.engineering_agent import EngineeringAgent  # pylint: disable=import-outside-toplevel
        
        self.agent = EngineeringAgent(pipeline_factory)
        self.logger = composer_logger.logger

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Execute engineering agent and update workflow state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with required tools
        """
        try:
            # Skip if no intent classification available
            if not state.intent_classification:
                return state

            intent = state.intent_classification
            user_id = getattr(state, 'user_id', None)
            
            if not user_id:
                raise NodeExecutionError("engineering_agent", Exception("User ID required for tool orchestration"))

            # Extract user query for context
            user_query = getattr(state.messages[-1], 'content', '') if state.messages else ""
            
            # Delegate to agent for tool orchestration
            tools = await self.agent.orchestrate_tools(intent, user_id, user_query)

            # Convert tools to AvailableTool format for state
            available_tools = [
                AvailableTool(
                    name=getattr(tool, 'name', 'unnamed_tool'),
                    description=getattr(tool, 'description', 'No description available'),
                    parameters=getattr(tool, 'parameters', {}),
                    tool_type="static" if hasattr(tool, '_static') else "dynamic"
                )
                for tool in tools if tool is not None
            ]
            
            state.required_tools = available_tools
            return state

        except Exception as e:
            self.logger.error(
                "Engineering agent node execution failed",
                extra={
                    "user_id": getattr(state, 'user_id', 'unknown'),
                    "error": str(e)
                }
            )
            
            # Continue with empty tools on error
            state.required_tools = []
            return state