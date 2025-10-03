"""
Tool Executor Node for LangGraph workflows.
Wraps LangChain v1.0 ToolNode for reliable tool execution within workflows.
"""

from typing import List

from langchain_core.tools import BaseTool
from langchain.agents import ToolNode

from models import LangChainMessage
from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger


class ToolExecutorNode:
    """
    Executes tool calls produced by the previous agent or tool node.

    Uses LangChain v1.0 ToolNode for reliable tool execution with proper error handling.
    """

    def __init__(self, tools: List[BaseTool]):
        """
        Initialize tool executor node.

        Args:
            tools: List of available tools for execution
        """
        self.tools = {tool.name: tool for tool in tools}
        # ToolNode in v1.0 supports handle_tool_errors parameter for better error handling
        self.tool_node = ToolNode(tools, handle_tool_errors=True)
        self.logger = composer_logger.logger

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Execute tool calls from the last message.

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
                return state

            self.logger.info(
                "Executing tool calls",
                user_id=getattr(state, "user_id", "unknown"),
                tool_count=len(last_message.tool_calls),
                tools=[call.get("name", "unknown") for call in last_message.tool_calls],
            )

            # Execute tools using LangChain v1.0 ToolNode
            tool_results = await self.tool_node.ainvoke({"messages": [last_message]})

            # Add tool results to state messages (v1.0 compatible)
            if "messages" in tool_results:
                state.messages.extend(tool_results["messages"])
            elif hasattr(tool_results, "messages"):
                # Handle alternative response format
                state.messages.extend(tool_results.messages)

            # Log tool execution completion
            self.logger.info(
                "Tool execution completed",
                user_id=getattr(state, "user_id", "unknown"),
                completed_tools=[
                    call.get("name", "unknown") for call in last_message.tool_calls
                ],
            )

            return state

        except Exception as e:
            self.logger.error(
                "Tool execution failed",
                user_id=getattr(state, "user_id", "unknown"),
                error=str(e),
                tools=list(self.tools.keys()),
            )

            # Add error message to state
            error_message = LangChainMessage(
                type="ai", content=f"Tool execution failed: {str(e)}"
            )
            state.messages.append(error_message)

            return state