"""
Server-side tool execution node for the IDE workflow graph.

This node intercepts tool calls for server-side tools (web_search, web_fetch)
from the agent's response and executes them locally, appending results back
to state so the agent can continue with the tool output.

Works with the Message-based WorkflowState (not LangChain AIMessage), making
it compatible with the existing agent/state architecture.
"""

from typing import Set

from graph.state import WorkflowState
from tools.server_tool_executor import (
    extract_server_tool_calls,
    execute_server_tool,
    _CLIENT_TOOL_NAME_MAP,
)
from models import MessageRole
from models.message import Message, MessageContent, MessageContentType
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ServerToolNode")


class ServerToolNode:
    """Graph node that executes server-side tool calls from the last assistant message.

    Only processes tool calls whose names match the provided server_tool_names set.
    Other tool calls (client-side) are left untouched for proxy passthrough.

    Populates ``state.server_tool_events`` with dicts of the form::

        {"tool_call": ToolCall, "result_text": str, "canonical_name": str}

    so the executor / router can emit the correct SSE content blocks.
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

        new_events: list[dict] = []
        result_parts: list[str] = []

        for tc in server_calls:
            result_text = await execute_server_tool(tc)
            canonical = _CLIENT_TOOL_NAME_MAP.get(tc.name, tc.name)

            new_events.append(
                {
                    "tool_call": tc,
                    "result_text": result_text,
                    "canonical_name": canonical,
                }
            )
            result_parts.append(result_text)

        # --- Rewrite messages so the local model understands tool results ---
        # Local models don't handle ToolMessage / tool_call_id well.
        # Instead:
        #   1. Strip server tool calls from the last assistant message
        #      (keep any text content + client-side tool calls).
        #   2. Inject a USER message containing the search/fetch results
        #      so the model sees them as normal context it can reason over.
        server_call_names = {tc.name for tc in server_calls}
        remaining_tool_calls = [
            tc
            for tc in (last_message.tool_calls or [])
            if tc.name not in server_call_names
        ]
        last_message.tool_calls = remaining_tool_calls or None

        # Ensure the assistant message has some text content (avoids empty AIMessage
        # which LangChain drops / the model sees as EOS).
        has_text = any(
            c.type == MessageContentType.TEXT and c.text for c in last_message.content
        )
        if not has_text:
            query_summaries = []
            for tc in server_calls:
                args = tc.args or {}
                q = args.get("query") or args.get("url") or tc.name
                query_summaries.append(q)
            last_message.content.append(
                MessageContent(
                    type=MessageContentType.TEXT,
                    text=f"[Performed server-side tool calls: {', '.join(query_summaries)}]",
                )
            )

        # Build a single USER message with all results
        combined_results = "\n\n".join(result_parts)
        state.messages.append(
            Message(
                role=MessageRole.USER,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=(
                            f"Here are the results from the tools you just invoked. "
                            f"Use these results to answer the original question:\n\n"
                            f"{combined_results}"
                        ),
                    )
                ],
            )
        )

        state.server_tool_events.extend(new_events)
        state.server_tool_iterations = 1  # reducer adds to existing count

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
