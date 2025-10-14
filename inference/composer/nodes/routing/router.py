"""
Workflow routing node for intelligent workflow selection and execution strategy.
Consolidates workflow-level routing logic from GraphBuilder into a dedicated, reusable component.
"""

from composer.graph.state import WorkflowState
from utils.logging import llmmllogger

# Inline workflow registry to avoid circular imports


class WorkflowRouter:
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
        self.user_id = user_id
        self.logger = llmmllogger.logger.bind(component="WorkflowRouter")

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
            i.workflow_type for i in state.intent_classification
        }

        return state
