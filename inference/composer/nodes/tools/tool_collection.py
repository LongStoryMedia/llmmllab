"""
Unified tool collection node that handles both static and dynamic tool collection.
Simplifies tool management by centralizing decisions about what tools are needed.
"""

from typing import List

from models import Tool
from composer.graph.state import WorkflowState
from composer.tools.registry import ToolRegistry
from utils.logging import llmmllogger
from composer.agents.engineering_agent import EngineeringAgent
from utils import extract_text_from_message


class ToolCollectionNode:
    """
    Unified node responsible for collecting all tools (static and dynamic) based on user queries.
    Centralizes tool decision logic and simplifies the tool collection workflow.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        engineering_agent: EngineeringAgent,
    ):
        self.tool_registry = tool_registry
        self.engineering_agent = engineering_agent
        self.logger = llmmllogger.bind(component="ToolCollectionNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Collect all tools (static and dynamic) based on user query.
        """
        try:
            assert state.user_id
            assert state.current_user_message
            assert state.user_config

            self.logger.info(
                "Collecting tools for workflow",
                user_id=state.user_id,
            )

            # Step 1: Collect all pre-loaded static tools
            available_static_tools = state.static_tools or []
            static_tools = await self._collect_static_tools(available_static_tools)

            self.logger.info(
                "Static tools collected",
                user_id=state.user_id,
                static_tool_count=len(static_tools),
                static_tool_names=[tool.name for tool in static_tools],
            )

            # Step 2: Decide if dynamic tools are needed and create them
            dynamic_tools = await self._collect_dynamic_tools(
                user_query=extract_text_from_message(state.current_user_message),
                user_id=state.user_id,
                static_tools=static_tools,
                user_config=state.user_config,
            )

            # Step 3: Update state with collected tools
            # Note: static_tools were already loaded by StaticToolLoadingNode
            # We only need to add dynamic_tools to available_tools and update static_tools with filtered set

            # Update static tools with filtered set (removing unneeded tools)
            state.static_tools = static_tools
            state.dynamic_tools = dynamic_tools

            # Clear available_tools and rebuild with filtered static tools + new dynamic tools
            all_tools = static_tools + dynamic_tools
            state.available_tools = all_tools

            self.logger.info(
                "Tool collection completed",
                user_id=state.user_id,
                total_tools=len(all_tools),
                static_tools=len(static_tools),
                dynamic_tools=len(dynamic_tools),
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
        Collect all available static tools.
        """
        try:
            # For now, we will include all static tools.
            # Filtering can be added here later if needed.
            return available_static_tools

        except Exception as e:
            self.logger.error(f"Static tool collection failed: {e}")
            return []

    async def _collect_dynamic_tools(
        self,
        user_query: str,
        user_id: str,
        static_tools: List[Tool],
        user_config,
    ) -> List[Tool]:
        """
        Decide if dynamic tools are needed and create them using the engineering agent.
        """
        try:
            # Check if dynamic tool generation is enabled
            if not self._should_generate_dynamic_tools(user_query, user_config):
                self.logger.info(
                    "Dynamic tool generation disabled or not needed based on query.",
                    user_id=user_id,
                )
                return []

            self.logger.info(
                "Generating dynamic tools based on user query.",
                user_id=user_id,
            )

            # Use engineering agent to generate dynamic tool specifications
            dynamic_tool_specs = (
                await self.engineering_agent.generate_dynamic_tool_specification(
                    user_query=user_query,
                    user_id=user_id,
                    static_tools=static_tools,
                )
            )

            # Convert DynamicTool specs to generic Tool instances for workflow state
            dynamic_tools = []
            for dt_spec in dynamic_tool_specs:
                tool = Tool(
                    name=dt_spec.name,
                    description=dt_spec.description,
                    args_schema=dt_spec.args_schema,
                    return_direct=dt_spec.return_direct,
                    tags=dt_spec.tags,
                    metadata=dt_spec.metadata,
                    handle_tool_error=dt_spec.handle_tool_error,
                    handle_validation_error=dt_spec.handle_validation_error,
                    response_format=dt_spec.response_format,
                )
                dynamic_tools.append(tool)

                # Register with tool registry for potential future reuse
                await self.tool_registry.register_dynamic_tool_instance(
                    tool_id=f"{user_id}_{dt_spec.name}",
                    tool_instance=tool,
                    user_id=user_id,
                )

            return dynamic_tools

        except Exception as e:
            self.logger.error(f"Dynamic tool collection failed: {e}")
            return []

    def _should_generate_dynamic_tools(
        self, user_query: str, user_config
    ) -> bool:
        """
        Determine if dynamic tools should be generated based on user query and configuration.
        """
        # Check user configuration first
        if (
            user_config
            and user_config.tool
            and not user_config.tool.enable_tool_generation
        ):
            return False

        # Simple keyword-based heuristic to trigger tool generation
        trigger_keywords = [
            "create a tool",
            "make a function",
            "write a script",
            "generate code to",
            "new tool for",
        ]
        if any(keyword in user_query.lower() for keyword in trigger_keywords):
            return True

        return False

    def _should_include_static_tool(
        self,
        tool: Tool,
        user_query: str,
    ) -> bool:
        """
        Determine if a static tool should be included based on the user query.
        """
        # This is a placeholder for more sophisticated logic.
        # For now, we can include tools based on keywords in the query.
        tool_name = getattr(tool, "name", "").lower()
        query_lower = user_query.lower()

        if "search" in query_lower and "search" in tool_name:
            return True
        if "memory" in query_lower and "memory" in tool_name:
            return True

        # Include all tools for now to ensure functionality.
        return True

    def _needs_basic_tools(self, user_query: str) -> bool:
        """
        Determine if basic tools are needed for simple requests.
        """
        # This is a placeholder for more sophisticated logic.
        return True
