"""
Unified tool collection node that handles both static and dynamic tool collection.
Simplifies tool management by centralizing decisions about what tools are needed.
"""

from typing import List

from models import Tool, IntentAnalysis
from composer.graph.state import WorkflowState
from composer.tools.registry import ToolRegistry
from composer.utils.extraction import extract_content_from_langchain_message
from composer.agents.engineering_agent import EngineeringAgent
from utils.logging import llmmllogger


class ToolCollectionNode:
    """
    Unified node responsible for collecting all tools (static and dynamic) based on user queries and intent analysis.
    Centralizes tool decision logic and simplifies the tool collection workflow.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        engineering_agent: EngineeringAgent,
    ):
        self.tool_registry = tool_registry
        self.engineering_agent = engineering_agent
        self.logger = llmmllogger.logger.bind(component="ToolCollectionNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Collect all tools (static and dynamic) based on user query and intent analysis.
        """
        try:
            assert state.user_id
            assert state.intent_classification
            assert state.current_user_message
            assert state.user_config

            self.logger.info(
                "Collecting tools for workflow",
                user_id=state.user_id,
                intent_count=len(state.intent_classification),
            )

            # Step 1: Collect static tools
            static_tools = await self._collect_static_tools(
                state.user_id, state.intent_classification, state.user_config
            )
            
            self.logger.info(
                "Static tools collected",
                user_id=state.user_id,
                static_tool_count=len(static_tools),
                static_tool_names=[tool.name for tool in static_tools],
            )

            # Step 2: Decide if dynamic tools are needed and create them
            dynamic_tools = await self._collect_dynamic_tools(
                user_query=extract_content_from_langchain_message(
                    state.current_user_message
                ),
                user_id=state.user_id,
                intents=state.intent_classification,
                static_tools=static_tools,
                user_config=state.user_config,
            )

            # Step 3: Update state with all collected tools
            all_tools = static_tools + dynamic_tools
            state.available_tools.extend(all_tools)
            state.static_tools = static_tools
            state.dynamic_tools = dynamic_tools

            self.logger.info(
                "Tool collection completed",
                user_id=state.user_id,
                total_tools=len(all_tools),
                static_tools=len(static_tools),
                dynamic_tools=len(dynamic_tools),
            )

        except Exception as e:
            self.logger.error(f"Tool collection failed: {e}")

        return state

    async def _collect_static_tools(
        self, user_id: str, _intents: List[IntentAnalysis], _user_config
    ) -> List[Tool]:
        """
        Collect static tools based on intent analysis and user configuration.
        """
        try:
            # Get all available static tools from registry
            available_static_tools = await self.tool_registry.get_static_tool_instances(user_id)
            
            # For now, include all static tools - can add filtering logic later based on intent
            # This is simpler and more predictable than complex intent-based filtering
            static_tools = available_static_tools
            
            # Could add intent-based filtering here if needed:
            # for intent in intents:
            #     if self._should_include_tool_for_intent(tool, intent):
            #         static_tools.append(tool)
            
            return static_tools
            
        except Exception as e:
            self.logger.error(f"Static tool collection failed: {e}")
            return []

    async def _collect_dynamic_tools(
        self,
        user_query: str,
        user_id: str,
        intents: List[IntentAnalysis],
        static_tools: List[Tool],
        user_config,
    ) -> List[Tool]:
        """
        Decide if dynamic tools are needed and create them using the engineering agent.
        """
        try:
            # Check if dynamic tool generation is enabled
            if not self._should_generate_dynamic_tools(intents, user_config):
                self.logger.info(
                    "Dynamic tool generation disabled or not needed",
                    user_id=user_id,
                )
                return []

            self.logger.info(
                "Generating dynamic tools",
                user_id=user_id,
            )

            # Use engineering agent to generate dynamic tool specifications
            dynamic_tool_specs = await self.engineering_agent.generate_dynamic_tool_specification(
                user_query=user_query,
                user_id=user_id,
                intents=intents,
                static_tools=static_tools,
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

    def _should_generate_dynamic_tools(self, intents: List[IntentAnalysis], user_config) -> bool:
        """
        Determine if dynamic tools should be generated based on intent and user configuration.
        """
        # Check user configuration
        if user_config and user_config.tool and not user_config.tool.enable_tool_generation:
            return False

        # Check if any intent requires tools or has high complexity
        for intent in intents:
            if (
                getattr(intent, "requires_tools", False)
                or getattr(intent, "complexity_level", "").lower() in ["high", "complex"]
            ):
                return True

        return False