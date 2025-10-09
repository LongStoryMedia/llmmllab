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

    def __init__(self, tool_registry=None):
        """
        Initialize tool executor node.

        Args:
            tool_registry: Registry containing executable tool instances
        """
        self.tool_registry = tool_registry
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
                self.logger.info(
                    "No tool calls found on last message - skipping execution",
                    user_id=getattr(state, "user_id", "unknown"),
                    last_message_type=getattr(last_message, "type", "unknown"),
                )
                return state

            # Get executable tools from registry instead of state
            executable_tools = {}
            if self.tool_registry:
                executable_tools = self.tool_registry.get_all_executable_tools()
            
            if not executable_tools:
                msg = "No executable tools available from registry"
                self.logger.error(
                    msg,
                    user_id=getattr(state, "user_id", "unknown"),
                    registry_available=bool(self.tool_registry),
                )
                raise RuntimeError(msg)

            self.logger.info(
                "Executing tool calls",
                user_id=getattr(state, "user_id", "unknown"),
                tool_count=len(last_message.tool_calls),
                tools=[call.get("name", "unknown") for call in last_message.tool_calls],
                available_tool_count=len(executable_tools),
            )

            # Use executable tools from registry
            name_to_tool = executable_tools  # This is already a name->BaseTool mapping
            self.logger.info(
                "Available tools debugging",
                user_id=getattr(state, "user_id", "unknown"),
                available_tools=list(name_to_tool.keys())[:10],  # Show first 10 for debugging
                total_available=len(name_to_tool),
                raw_tool_classes=[type(t).__name__ for t in name_to_tool.values()][:5],  # Show class names for debugging
                raw_tool_names=[getattr(t, 'name', 'NO_NAME') for t in name_to_tool.values()][:5],  # Show .name attrs
            )
            for call in last_message.tool_calls:
                tool_name = call.get("name")
                args = call.get("args") or call.get("arguments") or {}
                if tool_name not in name_to_tool:
                    self.logger.error(
                        "Tool not found debugging",
                        user_id=getattr(state, "user_id", "unknown"),
                        requested_tool=tool_name,
                        available_tools=sorted(list(name_to_tool.keys())),
                    )
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
                            run_fn = getattr(tool, "_run", None) or getattr(
                                tool, "run", None
                            )
                            if run_fn is None:
                                raise RuntimeError(
                                    f"Tool '{tool_name}' has no runnable method"
                                )
                            result = run_fn(**args)
                except Exception as te:
                    raise RuntimeError(
                        f"Tool '{tool_name}' execution failed: {te}"
                    ) from te

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
