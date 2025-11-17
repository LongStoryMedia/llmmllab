"""
Unified tool collection node that handles both static and dynamic tool collection.
Simplifies tool management by centralizing decisions about what tools are needed.
"""

from typing import List

from models import Tool, IntentAnalysis
from composer.graph.state import WorkflowState
from composer.tools.registry import ToolRegistry
from utils.logging import llmmllogger


class ToolCollectionNode:
    """
    Unified node responsible for collecting all tools (static and dynamic) based on user queries and intent analysis.
    Centralizes tool decision logic and simplifies the tool collection workflow.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
    ):
        self.tool_registry = tool_registry
        self.logger = llmmllogger.logger.bind(component="ToolCollectionNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Collect all tools (static and dynamic) based on user query and intent analysis.
        """
        try:
            assert state.user_id
            # assert state.intent_classification
            assert state.current_user_message
            assert state.user_config

            self.logger.info(
                "Collecting tools for workflow",
                user_id=state.user_id,
            )

            # Step 1: Filter pre-loaded static tools based on intent
            # Static tools should already be loaded by StaticToolLoadingNode
            available_static_tools = state.static_tools or []
            static_tools = await self._collect_static_tools(available_static_tools)

            self.logger.info(
                "Static tools collected",
                user_id=state.user_id,
                static_tool_count=len(static_tools),
                static_tool_names=[tool.name for tool in static_tools],
            )

            # Dynamic tool generation removed; agent can invoke create_dynamic_tool itself.
            state.static_tools = static_tools
            state.dynamic_tools = []
            state.available_tools = list(static_tools)

            self.logger.info(
                "Tool collection completed (static only)",
                user_id=state.user_id,
                total_tools=len(static_tools),
                static_tools=len(static_tools),
                dynamic_tools=0,
            )

        except Exception as e:
            self.logger.error(f"Tool collection failed: {e}", exc_info=True)
            raise

        return state

    async def _collect_static_tools(
        self,
        available_static_tools: List[Tool],
    ) -> List[Tool]:
        """
        Filter pre-loaded static tools based on intent analysis and user configuration.
        Uses intent-based filtering to select relevant static tools from the pre-loaded set.
        """
        try:
            # Apply intent-based filtering to pre-loaded static tools
            static_tools = []
            for tool in available_static_tools:
                static_tools.append(tool)

            return static_tools

        except Exception as e:
            self.logger.error(f"Static tool collection failed: {e}")
            return []

    # Dynamic tool generation removed; agent now uses create_dynamic_tool directly.

    def _should_include_static_tool(
        self,
        tool: Tool,
        intents: List[IntentAnalysis],
    ) -> bool:
        """
        Determine if a static tool should be included based on intent analysis.
        """
        tool_name = getattr(tool, "name", "").lower()

        for intent in intents:
            # Always include if intent explicitly requires tools
            if intent.requires_tools:
                # Convert required capabilities to values for comparison
                required_cap_values = [
                    cap.value for cap in intent.required_capabilities
                ]

                # Include search tools for information retrieval capabilities
                if (
                    "web_search" in required_cap_values
                    or "information_retrieval" in required_cap_values
                ) and "search" in tool_name:
                    return True

                # Include memory tools for conversation memory capabilities
                if (
                    "conversation_memory" in required_cap_values
                    and "memory" in tool_name
                ):
                    return True

                # Include processing tools for data/file manipulation
                processing_caps = [
                    "data_processing",
                    "file_manipulation",
                    "text_processing",
                ]
                if any(cap in required_cap_values for cap in processing_caps) and any(
                    keyword in tool_name for keyword in ["process", "file", "text"]
                ):
                    return True

                # Include API tools for integration capabilities
                if "api_integration" in required_cap_values and "api" in tool_name:
                    return True

                # Include basic math tools
                if "basic_math" in required_cap_values and any(
                    keyword in tool_name for keyword in ["math", "calc", "compute"]
                ):
                    return True

            # Include basic tools for moderate to high complexity
            if intent.complexity_level.value in [
                "MODERATE",
                "COMPLEX",
                "SPECIALIZED",
            ] and tool_name in ["web_search", "memory_search", "summarization"]:
                return True

        return False

    def _needs_basic_tools(self, intents: List[IntentAnalysis]) -> bool:
        """
        Determine if basic tools are needed for simple requests.
        """
        for intent in intents:
            # Convert required capabilities to values for comparison
            required_cap_values = [cap.value for cap in intent.required_capabilities]

            # Simple requests that benefit from basic tools
            basic_caps = ["information_retrieval", "web_search", "basic_math"]
            if intent.complexity_level.value in ["TRIVIAL", "SIMPLE"] and any(
                cap in required_cap_values for cap in basic_caps
            ):
                return True

        return False
