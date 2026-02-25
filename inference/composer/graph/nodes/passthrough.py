"""
Title generation node for conversation titles.
Generates concise, descriptive titles based on conversation content.
"""

from typing import List, Optional, Type
from pydantic import BaseModel

from langchain.agents.middleware import AgentMiddleware
from langchain_core.runnables import RunnableLambda

from composer.agents.chat import ChatAgent
from composer.graph.state import WorkflowState
from composer.graph.middleware.summarization_middleware import SummarizationMiddleware
from composer.constants import AGENT_NODE_NAME, STRUCTURED_AGENT_RUNNABLE_NAME

from models import NodeMetadata, Message, MessageRole
from utils.logging import llmmllogger


class PassthroughNode:
    """
    Generates a conversation title if none exists.

    Uses grammar-constrained LLM to generate concise, descriptive titles
    based on conversation content.
    """

    def __init__(
        self,
        agent: ChatAgent,
        node_metadata: NodeMetadata,
        grammar: Optional[Type[BaseModel]] = None,
    ):
        """
        Initialize title generation node with dependency injection.

        Args:
            agent: Required ClassifierAgent instance
        """
        self.agent = agent.bind_node_metadata(node_metadata)
        self.logger = llmmllogger.bind(component=AGENT_NODE_NAME)
        self.grammar = grammar

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Generate conversation title if needed.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with title
        """
        assert state.conversation_id is not None
        try:
            n_ctx = self.agent.profile.parameters.num_ctx or 100000
            max_tokens_before_summary = int(n_ctx * 0.95)
            middleware: List[AgentMiddleware] = [
                SummarizationMiddleware(
                    agent=self.agent,
                    max_tokens_before_summary=max_tokens_before_summary,
                    conversation_id=state.conversation_id,
                )
            ]

            if self.grammar:
                self.logger.info("Using structured output grammar for agent response")
                structured_response = await self.agent.run_structured(
                    message_input=state.messages,
                    middleware=middleware,
                    grammar=self.grammar,
                )

                runnable = RunnableLambda(
                    lambda x: x, name=STRUCTURED_AGENT_RUNNABLE_NAME
                )

                self.logger.debug(
                    f"Structured response from agent: {structured_response.model_dump_json(warnings=False)}"
                )

                runnable.invoke(structured_response)

                state.messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=[],
                        structured_output=structured_response.model_dump(
                            warnings=False
                        ),
                    )
                )
            else:
                response = await self.agent.run(
                    messages=state.messages,
                    middleware=middleware,
                )

                if response.message:
                    if response.message.tool_calls:
                        self.logger.info(
                            f"🔧 Generated {len(response.message.tool_calls)} tool calls"
                        )
                    state.messages.append(response.message)

            return state

        except Exception as e:
            self.logger.error(
                "Chat Agent failed",
                extra={
                    "user_id": getattr(state, "user_id", "unknown"),
                    "error": str(e),
                },
            )
            # Escalate by raising so tests fail visibly
            raise
