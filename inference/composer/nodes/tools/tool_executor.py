"""
Tool Executor Node for LangGraph workflows.
Uses the ToolsAgentSubgraph for efficient tool execution with minimal state overhead.
"""

from models import LangChainMessage
from composer.graph.state import WorkflowState
from composer.tools.registry import ToolRegistry
from composer.graph.subgraphs import ToolsAgentSubgraph
from utils.logging import llmmllogger


class ToolExecutorNode:
    """
    Executes tool calls using the ToolsAgentSubgraph for clean state isolation.

    This node delegates tool execution to a specialized subgraph that handles
    tools with minimal state overhead to prevent context window bloat.
    """

    def __init__(self, tool_registry: "ToolRegistry", chat_agent):
        """
        Initialize tool executor node.

        Args:
            tool_registry: Registry containing executable tool instances
            chat_agent: Chat agent for LLM operations in the subgraph
        """
        self.tool_registry = tool_registry
        self.chat_agent = chat_agent
        self.logger = llmmllogger.logger.bind(component="ToolExecutorNode")
        
        # Create the subgraph instance
        self.tools_agent_subgraph = ToolsAgentSubgraph(
            tool_registry=tool_registry,
            chat_agent=chat_agent,
        )

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Execute tool calls using the subgraph architecture.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with tool results
        """
        try:
            if not state.messages:
                return state

            last_message = state.messages[-1]

            # Check if last message has tool calls
            if not (hasattr(last_message, "tool_calls") and last_message.tool_calls):
                self.logger.info(
                    "No tool calls found on last message - skipping execution",
                    user_id=getattr(state, "user_id", "unknown"),
                    last_message_type=getattr(last_message, "type", "unknown"),
                )
                return state

            tool_calls = getattr(last_message, "tool_calls", [])
            self.logger.info(
                "Delegating tool execution to subgraph",
                user_id=getattr(state, "user_id", "unknown"),
                tool_count=len(tool_calls),
                tools=[call.get("name", "unknown") for call in tool_calls],
            )

            # Execute tools via subgraph
            self.logger.info(
                f"🔄 ToolExecutorNode: Calling tools_agent_subgraph.execute"
            )
            command = await self.tools_agent_subgraph.execute(state)

            # Apply the command updates to the state
            if command and command.update:
                self.logger.info(
                    f"🔄 ToolExecutorNode: Got command with {len(command.update)} updates: {list(command.update.keys())}"
                )
                if "messages" in command.update:
                    old_count = len(state.messages)
                    new_count = len(command.update["messages"])
                    self.logger.info(
                        f"🔄 ToolExecutorNode: Messages update - old: {old_count}, new: {new_count}"
                    )

                for key, value in command.update.items():
                    setattr(state, key, value)

                self.logger.info(
                    "Tool execution completed via subgraph",
                    user_id=getattr(state, "user_id", "unknown"),
                    completed_tools=[
                        call.get("name", "unknown") for call in last_message.tool_calls
                    ],
                    state_updates=list(command.update.keys()),
                )
            else:
                self.logger.warning(
                    "Tool execution returned no updates",
                    user_id=getattr(state, "user_id", "unknown"),
                )

            return state

        except Exception as e:
            self.logger.error(
                "Tool execution failed",
                user_id=getattr(state, "user_id", "unknown"),
                error=str(e),
            )
            raise
