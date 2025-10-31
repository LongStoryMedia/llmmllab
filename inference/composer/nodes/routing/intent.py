"""
Intent classification node for workflow routing.
Directly integrates the PlanningIntentSubgraph (class-based) without lazy global getter.
"""

from typing import TYPE_CHECKING

from composer.graph.state import WorkflowState
from composer.graph.subgraphs.planning_intent import PlanningIntentSubgraph
from composer.utils.state import assemble_context_messages
from utils.logging import llmmllogger

if TYPE_CHECKING:
    from composer.agents.classifier_agent import ClassifierAgent


class IntentClassifierNode:
    """LangGraph node wrapper for intent classification with planning middleware."""

    def __init__(self, classifier_agent: "ClassifierAgent"):
        self.agent = classifier_agent
        self.logger = llmmllogger.logger.bind(component="IntentClassifierNode")
        # Instantiate planning subgraph once; avoids import-based factory removed earlier
        self._planning_subgraph = PlanningIntentSubgraph(classifier_agent)

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        assert state.user_config
        assert state.user_id
        assert state.current_user_message
        if not state.messages:
            return state
        try:
            self.logger.info(
                "Intent classifier executing (planning subgraph)",
                extra={"user_id": state.user_id, "message_count": len(state.messages)},
            )
            command = await self._planning_subgraph.execute(state)
            if command and command.update:
                self.logger.debug(
                    "Applying planning subgraph updates",
                    update_keys=list(command.update.keys()),
                )
                for key, value in command.update.items():
                    if hasattr(state, key):
                        existing = getattr(state, key)
                        if isinstance(existing, list) and isinstance(value, list):
                            existing.extend(value)
                        else:
                            setattr(state, key, value)
                    else:
                        setattr(state, key, value)
            self.logger.info("Intent classification completed")
        except Exception as e:
            self.logger.error(
                "Intent classifier planning subgraph failed; falling back",
                extra={
                    "user_id": getattr(state, "user_id", "unknown"),
                    "error": str(e),
                },
            )
            try:
                intent_analyses = await self.agent.analyze(
                    messages=assemble_context_messages(state),
                    available_static_tools=state.static_tools,
                )
                state.intent_classification.extend(intent_analyses)
            except Exception as fallback_error:
                self.logger.error(
                    "Fallback classifier analyze failed", error=str(fallback_error)
                )
                raise e
        return state
