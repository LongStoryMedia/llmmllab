"""
Intent classification node for workflow routing.
Uses PlanningIntentSubgraph to provide sophisticated planning middleware.
"""

from typing import TYPE_CHECKING

from composer.graph.state import WorkflowState
from composer.graph.subgraphs.planning_intent import get_planning_intent_subgraph
from utils.logging import llmmllogger

if TYPE_CHECKING:
    from composer.agents.classifier_agent import ClassifierAgent


class IntentClassifierNode:
    """
    LangGraph node wrapper for intent classification with planning middleware.

    Uses PlanningIntentSubgraph to provide multi-step planning approach with
    todo list generation, complexity estimation, and tool requirement analysis.
    """

    def __init__(self, classifier_agent: "ClassifierAgent"):
        """
        Initialize intent classifier node with dependency injection.

        Args:
            classifier_agent: Required ClassifierAgent instance (passed to planning subgraph)
        """
        self.agent = classifier_agent
        self.logger = llmmllogger.logger.bind(component="IntentClassifierNode") 

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Execute intent classification using planning middleware subgraph.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with planning results and intent classification
        """
        assert state.user_config
        assert state.user_id
        assert state.current_user_message
        try:
            if not state.messages:
                return state

            self.logger.info(
                "Intent classifier node executing with planning middleware",
                extra={"user_id": state.user_id, "message_count": len(state.messages)},
            )

            # Use planning middleware subgraph for sophisticated intent analysis
            planning_subgraph = get_planning_intent_subgraph()
            command = await planning_subgraph.execute(state)

            # Apply updates from planning middleware
            if command and command.update:
                for key, value in command.update.items():
                    if hasattr(state, key):
                        if isinstance(getattr(state, key), list):
                            # For list attributes, extend rather than replace
                            getattr(state, key).extend(value)
                        else:
                            setattr(state, key, value)
                    else:
                        setattr(state, key, value)

            self.logger.info("Intent classification with planning completed")

        except Exception as e:
            self.logger.error(
                "Intent classifier node with planning failed",
                extra={
                    "user_id": getattr(state, "user_id", "unknown"),
                    "error": str(e),
                },
            )
            # Fallback to direct agent call
            self.logger.warning("Falling back to direct classifier agent call")
            try:
                from composer.utils.state import assemble_context_messages
                intent_analyses = await self.agent.analyze(
                    messages=assemble_context_messages(state),
                    available_static_tools=state.static_tools,
                )
                state.intent_classification.extend(intent_analyses)
            except Exception as fallback_error:
                self.logger.error(f"Fallback also failed: {fallback_error}")
                raise e

        return state
