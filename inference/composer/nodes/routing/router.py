"""
Workflow routing node for intelligent workflow selection and execution strategy.
Consolidates workflow-level routing logic from GraphBuilder into a dedicated, reusable component.
"""

from composer.graph.state import WorkflowState
from composer.nodes.base_node import BaseNode

# Inline workflow registry to avoid circular imports


class WorkflowRouter(BaseNode):
    """
    Intelligent router for workflow selection and execution strategy.

    Consolidates workflow-level routing logic that was previously duplicated
    across GraphBuilder methods. Provides both routing decisions and execution
    strategy determination based on intent analysis and complexity.
    """

    def __init__(self, user_id: str):
        """
        Initialize workflow router.

        Args:
            user_id: User identifier for logging and context
        """
        super().__init__("workflow_router", user_id=user_id)
        
    def _initialize_node(self, pipeline_factory=None, **kwargs) -> None:
        """Initialize WorkflowRouter with dependency injection."""
        self.user_id = kwargs.get('user_id')

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Route workflows and determine execution strategy.

        This is the main entry point that updates the state with routing decisions
        and execution strategy for the coordinator to use.

        Args:
            state: Current workflow state

        Returns:
            Updated state with routing decisions
        """
        state.selected_workflows = {
            i.primary_intent for i in state.intent_classification
        }

        return state
