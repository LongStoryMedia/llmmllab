"""
Intent classification node for workflow routing.
Wraps the IntentClassifierAgent to provide LangGraph workflow integration.
"""

from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError
from composer.utils.conversion import langchain_message_to_message


class IntentClassifierNode:
    """
    LangGraph node wrapper for intent classification.

    Wraps the IntentClassifierAgent to provide workflow state integration and
    proper LangGraph node interface. Handles state updates and RAG routing configuration.
    """

    def __init__(self):
        """
        Initialize intent classifier node with agent delegation.
        """
        # Lazy import to avoid circular dependencies
        from composer.agents.intent_classifier import (  # pylint: disable=import-outside-toplevel
            IntentClassifierAgent,
        )

        self.agent = IntentClassifierAgent()
        self.logger = composer_logger.logger

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Execute intent classification using the wrapped agent.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with intent classification and RAG config
        """
        try:
            user_id = getattr(state, "user_id", None)
            if not user_id:
                raise NodeExecutionError(
                    "intent_classifier",
                    Exception("User ID required for intent classification"),
                )

            if not state.messages:
                return state

            assert state.current_user_message

            self.logger.info(
                "Intent classifier node executing",
                extra={"user_id": user_id, "message_count": len(state.messages)},
            )

            # Convert WorkflowState messages to Message format expected by agent
            messages = []
            for msg in state.messages:
                # Convert LangChainMessage to Message format if needed
                messages.append(msg)

            # Delegate to the specialized intent classifier agent
            intent_analyses = await self.agent.analyze(
                user_id,
                langchain_message_to_message(state.current_user_message),
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
