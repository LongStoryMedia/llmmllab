"""Tool composition node (dynamic tools already persisted during creation)."""

from composer.graph.state import WorkflowState
from utils.logging import llmmllogger


class ToolComposerNode:
    """
    A node that composes multiple tools into a single workflow.
    """

    def __init__(self):
        """
        Initialize the tool composer node.
        """
        self.logger = llmmllogger.logger.bind(component="ToolComposerNode")

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

            # Dynamic tools have already been persisted at creation time; just include them
            for tool in state.dynamic_tools:
                tool_name = getattr(tool, "name", str(tool))
                if tool_name not in seen_names:
                    seen_names.add(tool_name)
                    deduplicated_tools.append(tool)

            state.available_tools = deduplicated_tools

            self.logger.info(
                "Tools deduplicated and optimized",
                user_id=state.user_id,
                final_count=len(deduplicated_tools),
            )

            return state

        except Exception as e:
            # Record error and re-raise so test framework captures failure
            self.logger.error(f"Tool composition failed: {e}")
            raise
