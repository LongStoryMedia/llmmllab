from typing import List, cast

from models import Tool, IntentAnalysis, DynamicTool
from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger


class ToolComposerNode:
    """
    A node that composes multiple tools into a single workflow.
    """

    def __init__(self):
        """
        Initialize the tool composer node.

        Args:
            tools: List of tools to compose
        """
        self.logger = composer_logger.logger.bind(component="ToolComposerNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Deduplicate and optimize the combined tool set.
        """
        try:
            assert state.intent_classification
            assert state.user_id

            self.logger.info(
                "Deduplicating and optimizing tools",
                user_id=state.user_id,
                static_count=len(state.static_tools or []),
                dynamic_count=len(state.dynamic_tools or []),
            )

            # Combine all tools
            all_tools = []
            all_tools.extend(state.static_tools or [])
            all_tools.extend(state.dynamic_tools or [])

            # Deduplicate by tool name
            seen_names = set()
            deduplicated_tools = []

            for tool in state.static_tools:
                tool_name = getattr(tool, "name", str(tool))
                if tool_name not in seen_names:
                    seen_names.add(tool_name)
                    deduplicated_tools.append(tool)

            from db import storage  # pylint: disable=import-outside-toplevel

            tool_svc = storage.get_service(storage.dynamic_tool)

            for tool in state.dynamic_tools:
                tool_name = getattr(tool, "name", str(tool))
                if tool_name not in seen_names:
                    seen_names.add(tool_name)
                    deduplicated_tools.append(tool)
                    await tool_svc.create_tool(cast(DynamicTool, tool))

            state.available_tools = deduplicated_tools

            self.logger.info(
                "Tools deduplicated and optimized",
                user_id=state.user_id,
                final_count=len(deduplicated_tools),
            )

            return state

        except Exception as e:
            self.logger.error(f"Tool composition failed: {e}")
            # Fallback to simple combination - ensure type consistency
            combined_tools = []
            combined_tools.extend(state.static_tools or [])
            combined_tools.extend(state.dynamic_tools or [])
            state.available_tools = combined_tools
            return state
