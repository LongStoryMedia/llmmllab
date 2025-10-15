"""
Intent classification node for workflow routing.
Wraps the ClassifierAgent to provide LangGraph workflow integration.
"""

from typing import TYPE_CHECKING


from composer.graph.state import WorkflowState
from composer.utils.conversion import convert_langchain_messages_to_messages
from utils.logging import llmmllogger

if TYPE_CHECKING:
    from composer.agents.classifier_agent import ClassifierAgent


class IntentClassifierNode:
    """
    LangGraph node wrapper for intent classification.

    Wraps the ClassifierAgent to provide workflow state integration and
    proper LangGraph node interface. Handles state updates and RAG routing configuration.
    """

    def __init__(self, classifier_agent: "ClassifierAgent"):
        """
        Initialize intent classifier node with dependency injection.

        Args:
            classifier_agent: Required ClassifierAgent instance
        """
        self.agent = classifier_agent
        self.logger = llmmllogger.logger.bind(component="IntentClassifierNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Execute intent classification using the wrapped agent.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with intent classification and RAG config
        """
        assert state.user_config
        assert state.user_id
        assert state.current_user_message
        try:
            if not state.messages:
                return state

            self.logger.info(
                "Intent classifier node executing",
                extra={"user_id": state.user_id, "message_count": len(state.messages)},
            )

            # Convert WorkflowState messages to Message format expected by agent
            messages = []
            for msg in state.messages:
                # Convert LangChainMessage to Message format if needed
                messages.append(msg)

            # Delegate to the specialized intent classifier agent
            intent_analyses = await self.agent.analyze(
                user_id=state.user_id,
                messages=convert_langchain_messages_to_messages(messages),
            )

            # Extend workflow state with analysis results (list reducer)
            state.intent_classification.extend(intent_analyses)

            self.logger.info("Intent classification completed")

        except Exception as e:
            self.logger.error(
                "Intent classifier node failed",
                extra={
                    "user_id": getattr(state, "user_id", "unknown"),
                    "error": str(e),
                },
            )
            raise

        return state
