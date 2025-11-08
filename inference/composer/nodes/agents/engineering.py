"""
Engineering Agent Node for LangGraph workflow integration.
Provides LangGraph node wrapper for technical engineering response generation.
"""

from typing import TYPE_CHECKING

from models import Message, MessageRole, MessageContent, MessageContentType

from composer.graph.state import WorkflowState, assemble_context_messages
from composer.core.errors import NodeExecutionError
from utils.message_conversion import extract_text_from_message
from utils.logging import llmmllogger


if TYPE_CHECKING:
    from composer.agents.engineering_agent import EngineeringAgent


class EngineeringAgentNode:
    """
    LangGraph node wrapper for Engineering Agent.

    Handles workflow state management and delegates business logic to EngineeringAgent
    for technical response generation. Focuses on engineering expertise rather than
    tool orchestration (which is handled by ToolOrchestrationSubgraph).
    """

    def __init__(
        self,
        engineering_agent: "EngineeringAgent",
    ):
        """
        Initialize engineering agent node with dependency injection.

        Args:
            engineering_agent: Required EngineeringAgent instance
        """
        self.agent = engineering_agent
        self.logger = llmmllogger.logger.bind(component="EngineeringAgentNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Execute engineering agent for technical response generation.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with engineering response
        """
        try:
            # Skip if no messages or intent classification
            if (
                not state.messages
                or not state.intent_classification
                or not state.current_user_message
            ):
                return state

            user_id = getattr(state, "user_id", None)
            if not user_id:
                raise NodeExecutionError(
                    "engineering_agent",
                    Exception("User ID required for engineering responses"),
                )

            # Debug logging for all intent classification data received
            self.logger.info(
                f"Engineering node received {len(state.intent_classification)} intent classifications",
                extra={
                    "user_id": user_id,
                    "intent_count": len(state.intent_classification),
                },
            )

            for i, intent in enumerate(state.intent_classification):
                # Debug logging for each intent
                self.logger.info(
                    f"Engineering intent {i+1}: workflow_type={intent.workflow_type}, "
                    f"technical_domain={intent.technical_domain}, "
                    f"response_format={intent.response_format}",
                    extra={
                        "user_id": user_id,
                        "intent_index": i,
                        "workflow_type": (
                            str(intent.workflow_type) if intent.workflow_type else None
                        ),
                        "technical_domain": (
                            str(intent.technical_domain)
                            if intent.technical_domain
                            else None
                        ),
                        "response_format": (
                            str(intent.response_format)
                            if intent.response_format
                            else None
                        ),
                        "intent_object_type": type(intent).__name__,
                    },
                )

                response = await self.agent.generate_technical_response(
                    messages=assemble_context_messages(state),
                    user_id=user_id,
                    domain=intent.technical_domain,
                    response_format=intent.response_format,
                )

                if not response or not response.message:
                    self.logger.warning("Engineering agent returned no response")
                    continue

                # Add to messages with proper reducer handling
                state.messages.append(response.message)

                self.logger.info("Engineering response generated successfully")

            # Cleanup agent resources after completion
            self.agent.cleanup()

            return state

        except Exception as e:
            self.logger.error(
                "Engineering agent node execution failed",
                extra={
                    "user_id": getattr(state, "user_id", "unknown"),
                    "error": str(e),
                },
            )

            # Cleanup agent resources even on error
            self.agent.cleanup()

            # Continue workflow execution on error without adding response
            return state
