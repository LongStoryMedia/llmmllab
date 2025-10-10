"""
Tool Executor Node for LangGraph workflows.
Wraps LangChain v1.0 ToolNode for reliable tool execution within workflows.
"""

import json
from typing import Any, Dict

from models import LangChainMessage, SearchResult
from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger
from composer.tools.registry import ToolRegistry


class ToolExecutorNode:
    """
    Executes tool calls produced by the previous agent or tool node.

    Executes tool calls directly without relying on LangChain ToolNode (removed dependency).
    """

    def __init__(self, tool_registry: "ToolRegistry"):
        """
        Initialize tool executor node.

        Args:
            tool_registry: Registry containing executable tool instances
        """
        self.tool_registry = tool_registry
        self.logger = composer_logger.logger.bind(component="ToolExecutorNode")

    def _convert_to_search_result(self, search_data: Dict[str, Any], args: Dict[str, Any]) -> SearchResult:
        """
        Convert web search tool output to SearchResult format.
        
        Args:
            search_data: Parsed JSON result from web search tool
            args: Original tool arguments
            
        Returns:
            SearchResult instance
        """
        from models.search_result_content import SearchResultContent
        
        try:
            contents = []
            for item in search_data.get("results", []):
                content = SearchResultContent(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    content=item.get("content", ""),
                    relevance=item.get("relevance", 1.0)
                )
                contents.append(content)
            
            return SearchResult(
                is_from_url_in_user_query=False,  # Web search results are not from user URLs
                query=search_data.get("query", args.get("query", "")),
                contents=contents,
                error=search_data.get("error")
            )
        except Exception as e:
            self.logger.error(
                f"Failed to convert search data to SearchResult: {e}",
                search_data_keys=list(search_data.keys()) if search_data else []
            )
            return SearchResult(
                is_from_url_in_user_query=False,
                query=args.get("query", ""),
                contents=[],
                error=f"Conversion failed: {e}"
            )

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
                available_tools=list(name_to_tool.keys())[
                    :10
                ],  # Show first 10 for debugging
                total_available=len(name_to_tool),
                raw_tool_classes=[type(t).__name__ for t in name_to_tool.values()][
                    :5
                ],  # Show class names for debugging
                raw_tool_names=[
                    getattr(t, "name", "NO_NAME") for t in name_to_tool.values()
                ][
                    :5
                ],  # Show .name attrs
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

                # Handle special case for web search tool - extract search results to state
                if tool_name == "web_search" and isinstance(result, str):
                    try:
                        search_data = json.loads(result)
                        if search_data.get("status") == "success" and "results" in search_data:
                            # Convert search results to SearchResult format and add to state
                            search_result = self._convert_to_search_result(search_data, args)
                            if search_result:
                                state.web_search_results.append(search_result)
                                state.search_query = search_data.get("query") or args.get("query", "")
                                self.logger.info(
                                    "Added search results to state",
                                    user_id=getattr(state, "user_id", "unknown"),
                                    result_count=len(search_result.contents or []),
                                    query=state.search_query[:100] if state.search_query else "unknown"
                                )
                    except (json.JSONDecodeError, KeyError) as e:
                        self.logger.warning(
                            "Failed to extract search results from web_search tool",
                            user_id=getattr(state, "user_id", "unknown"),
                            error=str(e)
                        )

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
            raise
