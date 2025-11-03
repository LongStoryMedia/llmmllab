"""
Engineering Agent Node for LangGraph workflow integration.
Provides LangGraph node wrapper for technical engineering response generation.
"""

from typing import TYPE_CHECKING

from models import LangChainMessage, TechnicalDomain, ResponseFormat

from composer.graph.state import WorkflowState
from composer.core.errors import NodeExecutionError
from composer.utils.extraction import extract_content_from_langchain_message
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

            # Extract user query from last message using langgraph utility
            last_message = state.current_user_message
            user_query = extract_content_from_langchain_message(last_message)

            if not user_query.strip():
                return state

            for intent in state.intent_classification:
                # Use technical domain and response format from intent analysis
                domain = intent.technical_domain
                response_format = intent.response_format
                
                # Skip if engineering fields are not populated (shouldn't happen with fixed prompt)
                if not domain or not response_format:
                    self.logger.warning(
                        "Engineering intent missing required fields",
                        extra={
                            "user_id": user_id,
                            "intent_domain": intent.technical_domain,
                            "intent_format": intent.response_format,
                        },
                    )
                    continue
                
                self.logger.info(
                    "Engineering intent details",
                    extra={
                        "user_id": user_id, 
                        "intent_domain": intent.technical_domain,
                        "intent_format": intent.response_format,
                        "using_domain": domain,
                        "using_format": response_format,
                    },
                )
                self.logger.info(
                    "Generating engineering response",
                    extra={
                        "user_id": user_id,
                        "domain": domain,
                        "format": response_format,
                        "query_length": len(user_query),
                    },
                )

                # Generate technical response using engineering agent
                # Note: Engineering agent doesn't typically use tools in current implementation
                response = await self.agent.generate_technical_response(
                    query=user_query,
                    user_id=user_id,
                    domain=domain,
                    response_format=response_format,
                    tools=None,  # Engineering agent focuses on technical analysis, not tool execution
                    grammar=None,  # Could be enhanced based on state requirements
                )

                # Add engineering response to state messages

                engineering_response = LangChainMessage(
                    content=response,
                    additional_kwargs={
                        "agent": "engineering",
                        "domain": str(domain),
                        "format": str(response_format),
                    },
                )

                # Add to messages with proper reducer handling
                state.messages.append(engineering_response)

                self.logger.info(
                    "Engineering response generated successfully",
                    extra={
                        "user_id": user_id,
                        "response_length": len(response),
                        "domain": domain,
                    },
                )

            return state

        except Exception as e:
            self.logger.error(
                "Engineering agent node execution failed",
                extra={
                    "user_id": getattr(state, "user_id", "unknown"),
                    "error": str(e),
                },
            )

            # Continue workflow execution on error without adding response
            return state
