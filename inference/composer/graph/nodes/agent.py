"""
Title generation node for conversation titles.
Generates concise, descriptive titles based on conversation content.
"""

from typing import List
from langchain.agents.middleware import AgentMiddleware

from composer.tools.registry import ToolRegistry
from composer.agents.chat import ChatAgent
from composer.graph.state import WorkflowState
from composer.graph.middleware.summarization_middleware import SummarizationMiddleware

from models import NodeMetadata
from utils.logging import llmmllogger


class AgentNode:
    """
    Generates a conversation title if none exists.

    Uses grammar-constrained LLM to generate concise, descriptive titles
    based on conversation content.
    """

    def __init__(
        self, agent: ChatAgent, tool_registry: ToolRegistry, node_metadata: NodeMetadata
    ):
        """
        Initialize title generation node with dependency injection.

        Args:
            agent: Required ClassifierAgent instance
        """
        self.agent = agent.bind_node_metadata(node_metadata)
        self.logger = llmmllogger.bind(component="AgentNode")
        self.tool_registry = tool_registry

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
            tools = self.tool_registry.get_all_executable_tools()

            response = await self.agent.run(
                messages=state.messages,
                tools=tools,
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
