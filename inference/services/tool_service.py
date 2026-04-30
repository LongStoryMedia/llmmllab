"""
ToolService — shared tool preparation and server-tool separation.

Centralises the logic for splitting client-provided tool definitions into
tools that the *client* should execute vs. tools that the *server* executes
locally (e.g. web_search, web_fetch).  Both the Anthropic and OpenAI
streaming paths need this.
"""

from dataclasses import dataclass

from tools.server_tool_executor import (
    separate_server_tools,
    get_server_tool_names,
    make_server_tool_definitions,
    find_locally_executable_tools,
)
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="tool_service")


@dataclass
class PreparedTools:
    """Result of preparing client tools — ready to pass to the composer."""

    client_tools: list | None
    """Tool definitions to send to the model (client + server defs)."""

    server_tool_names: set[str]
    """Names of tools that should be intercepted and run server-side."""


class ToolService:
    """Stateless helpers for tool preparation."""

    @staticmethod
    def prepare_tools(client_tools: list | None) -> PreparedTools:
        """Separate server-side tools from client tools and return the
        combined definitions list plus the set of server-tool names.

        If *client_tools* is ``None`` or empty, returns an empty result
        with no server tools.
        """
        server_tool_names: set[str] = set()

        if not client_tools:
            return PreparedTools(client_tools=client_tools, server_tool_names=set())

        only_client, server_tools = separate_server_tools(client_tools)
        if server_tools:
            server_tool_names = get_server_tool_names(server_tools)
            server_defs = make_server_tool_definitions(server_tools)
            client_tools = only_client + server_defs
            logger.info(
                "Separated server-side tools for local execution",
                extra={
                    "server_tools": list(server_tool_names),
                    "client_tool_count": len(only_client),
                    "server_def_count": len(server_defs),
                },
            )

        # Also detect client tools like WebSearch/WebFetch that should be
        # executed locally.
        local_names = find_locally_executable_tools(client_tools)
        if local_names:
            server_tool_names |= local_names
            logger.info(
                "Detected locally-executable client tools",
                extra={"local_tools": list(local_names)},
            )

        return PreparedTools(
            client_tools=client_tools,
            server_tool_names=server_tool_names,
        )
