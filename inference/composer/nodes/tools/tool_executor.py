"""
Tool Executor Node for LangGraph workflows.
Wraps LangChain v1.0 ToolNode for reliable tool execution within workflows.
"""

from typing import List, cast

from langchain_core.tools import BaseTool

from models import LangChainMessage
from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger


class ToolExecutorNode:
    """
    Executes tool calls produced by the previous agent or tool node.

    Executes tool calls directly without relying on LangChain ToolNode (removed dependency).
    """

    def __init__(self):
        """
        Initialize tool executor node.

        Args:
            tools: List of available tools for execution. If empty, will use state.required_tools at runtime.
        """
        self.logger = composer_logger.logger.bind(component="ToolExecutorNode")

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

            tools_to_use: List[BaseTool] = []
            if state.available_tools:
                for tool in state.available_tools:
                    tools_to_use.append(cast(BaseTool, tool))

            if not tools_to_use:
                msg = "No compatible tools available for execution"
                self.logger.error(
                    msg,
                    user_id=getattr(state, "user_id", "unknown"),
                    state_tools=len(getattr(state, "available_tools", []) or []),
                )
                raise RuntimeError(msg)

            self.logger.info(
                "Executing tool calls",
                user_id=getattr(state, "user_id", "unknown"),
                tool_count=len(last_message.tool_calls),
                tools=[call.get("name", "unknown") for call in last_message.tool_calls],
                available_tool_count=len(tools_to_use),
            )

            # Direct execution: each tool call maps name->tool; pass arguments if present
            name_to_tool = {t.name: t for t in tools_to_use}
            for call in last_message.tool_calls:
                tool_name = call.get("name")
                args = call.get("args") or call.get("arguments") or {}
                if tool_name not in name_to_tool:
                    raise RuntimeError(f"Requested tool '{tool_name}' not available")
                tool = name_to_tool[tool_name]
                try:
                    if hasattr(tool, "_arun"):
                        result = await tool._arun(**args)  # type: ignore
                    else:
                        # Some community tools expose arun
                        arun = getattr(tool, "arun", None)
                        if arun:
                            result = await arun(**args)
                        else:
                            # Fallback to sync _run executed in threadpool? For now invoke directly
                            run_fn = getattr(tool, "_run", None) or getattr(tool, "run", None)
                            if run_fn is None:
                                raise RuntimeError(f"Tool '{tool_name}' has no runnable method")
                            result = run_fn(**args)
                except Exception as te:
                    raise RuntimeError(f"Tool '{tool_name}' execution failed: {te}") from te

                # Append tool result as assistant message for downstream consumption
                tool_message = LangChainMessage(type="tool", content=str(result))
                state.messages.append(tool_message)

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
            )
            # Record failure in execution metadata if available then raise
            if hasattr(state, "execution_metadata"):
                state.execution_metadata.add_error(f"Tool execution failed: {e}")
            raise
