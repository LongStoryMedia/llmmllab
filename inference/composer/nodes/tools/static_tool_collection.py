from composer.graph.state import WorkflowState
from composer.tools.registry import ToolRegistry
from composer.monitoring.logging import composer_logger


class StaticToolCollectionNode:
    """
    Node to collect static tools based on intent analysis.
    """

    def __init__(self, tool_registry: ToolRegistry):
        """
        Initialize the static tool collection node.

        Args:
            tool_registry: Registry to fetch static tools from
        """
        self.tool_registry = tool_registry
        self.logger = composer_logger.logger.bind(component="StaticToolCollectionNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Collect static tools based on intent analysis.
        """
        try:
            assert state.user_id
            assert state.intent_classification
            intent = state.intent_classification

            self.logger.info(
                "Collecting static tools",
                user_id=state.user_id,
            )
            static_tools = []

            for intent in state.intent_classification:
                # Get static tools from registry based on intent
                tools_for_intent = await self.tool_registry.get_tools_for_context(
                    intent, state.user_id
                )
                static_tools.extend(tools_for_intent)

            state.available_tools.extend(static_tools)
            state.static_tools = static_tools

            self.logger.info(
                "Static tools collected",
                user_id=state.user_id,
                static_tool_count=len(static_tools),
            )

        except Exception as e:
            self.logger.error(f"Static tool collection failed: {e}")

        return state
