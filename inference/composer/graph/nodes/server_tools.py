"""
Server-side tool execution node for the IDE workflow graph.

This node intercepts tool calls for server-side tools (web_search, web_fetch)
from the agent's response and executes them locally, appending results back
to state so the agent can continue with the tool output.

Works with the Message-based WorkflowState (not LangChain AIMessage), making
it compatible with the existing agent/state architecture.
"""

from typing import Set

from composer.graph.state import WorkflowState
from composer.tools.server_tool_executor import (
    extract_server_tool_calls,
    execute_server_tool_calls,
)
from models import MessageRole
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ServerToolNode")


class ServerToolNode:
    """Graph node that executes server-side tool calls from the last assistant message.

    Only processes tool calls whose names match the provided server_tool_names set.
    Other tool calls (client-side) are left untouched for proxy passthrough.
    """

    def __init__(self, server_tool_names: Set[str]):
        self.server_tool_names = server_tool_names

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        if not state.messages:
            return state

        last_message = state.messages[-1]
        if last_message.role != MessageRole.ASSISTANT or not last_message.tool_calls:
            return state

        server_calls, _client_calls = extract_server_tool_calls(
            last_message.tool_calls, self.server_tool_names
        )

        if not server_calls:
            return state

        logger.info(
            "Executing server-side tool calls",
            extra={
                "tool_names": [tc.name for tc in server_calls],
                "count": len(server_calls),
            },
        )

        tool_result_messages = await execute_server_tool_calls(server_calls)
        state.messages.extend(tool_result_messages)

        return state


def make_should_continue_server_tools(server_tool_names: Set[str]):
    """Create a routing function that routes to the server tool node only when
    the last message contains server-side tool calls.

    Returns "server_tools" if there are server tool calls, "end" otherwise.
    Client-only tool calls also route to "end" since they are proxied back.
    """

    def should_continue(state: WorkflowState) -> str:
        if not state.messages:
            return "end"

        last_message = state.messages[-1]
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return "end"

        # Check if any tool calls are for server-side tools
        for tc in last_message.tool_calls:
            if tc.name in server_tool_names:
                return "server_tools"

        return "end"

    return should_continue
