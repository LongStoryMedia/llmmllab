"""
Static tool loading node for early tool collection in the workflow.
Loads static tools and previously generated dynamic tools before intent analysis.
"""

from typing import List

from models import Tool
from composer.graph.state import WorkflowState
from composer.tools.registry import ToolRegistry
from utils.logging import llmmllogger


class StaticToolLoadingNode:
    """
    Node responsible for loading static tools and previously generated dynamic tools early in the workflow.
    Ensures static tools are available before intent analysis for better decision making.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        dynamic_tool_storage,
    ):
        self.tool_registry = tool_registry
        self.dynamic_tool_storage = dynamic_tool_storage
        self.logger = llmmllogger.logger.bind(component="StaticToolLoadingNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Load static tools and previously generated dynamic tools into workflow state.
        """
        try:
            assert state.user_id
            assert state.user_config

            self.logger.info(
                "Loading static tools and previous dynamic tools",
                user_id=state.user_id,
            )

            # Step 1: Load standard static tools
            static_tools = await self.tool_registry.get_static_tool_instances(
                state.user_id
            )

            # Step 2: Load previously generated dynamic tools for this user
            previous_dynamic_tools = await self._load_previous_dynamic_tools(
                state.user_id
            )

            # Step 3: Combine into unified static tool set (previous dynamic tools become static)
            all_static_tools = static_tools + previous_dynamic_tools

            # Step 4: Update workflow state with combined static tools
            state.static_tools = all_static_tools
            state.available_tools.extend(all_static_tools)

            self.logger.info(
                "Static tool loading completed",
                user_id=state.user_id,
                standard_static_tools=len(static_tools),
                previous_dynamic_tools=len(previous_dynamic_tools),
                total_static_tools=len(all_static_tools),
                static_tool_names=[tool.name for tool in all_static_tools],
            )

        except Exception as e:
            self.logger.error(f"Static tool loading failed: {e}")

        return state

    async def _load_previous_dynamic_tools(self, user_id: str) -> List[Tool]:
        """
        Load previously generated dynamic tools from storage and convert them to static tools.
        """
        try:
            # Get all previously generated dynamic tools for this user
            dynamic_tools, _ = await self.dynamic_tool_storage.list_tools_by_user(
                user_id=user_id, limit=100, offset=0  # Load up to 100 previous tools
            )

            # Convert DynamicTool instances to generic Tool instances for reuse as static tools
            previous_tools = []
            for dt in dynamic_tools:
                tool = Tool(
                    name=dt.name,
                    description=dt.description,
                    args_schema=dt.args_schema,
                    return_direct=dt.return_direct,
                    tags=dt.tags,
                    metadata=dt.metadata,
                    handle_tool_error=dt.handle_tool_error,
                    handle_validation_error=dt.handle_validation_error,
                    response_format=dt.response_format,
                )
                previous_tools.append(tool)

            self.logger.info(
                "Previous dynamic tools loaded as static tools",
                user_id=user_id,
                count=len(previous_tools),
                tool_names=[tool.name for tool in previous_tools],
            )

            return previous_tools

        except Exception as e:
            self.logger.error(f"Failed to load previous dynamic tools: {e}")
            return []
