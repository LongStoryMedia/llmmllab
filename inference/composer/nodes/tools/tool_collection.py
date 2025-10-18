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

            # Step 1: Filter pre-loaded static tools based on intent
            # Static tools should already be loaded by StaticToolLoadingNode
            available_static_tools = state.static_tools or []
            static_tools = await self._collect_static_tools(
                state.intent_classification,
                available_static_tools,
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
            self.logger.error(f"Tool collection failed: {e}")

        return state

    async def _collect_static_tools(
        self,
        intents: List[IntentAnalysis],
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
                if self._should_include_static_tool(tool, intents):
                    static_tools.append(tool)

            # If no tools match intent filtering, fall back to basic tools for simple requests
            if not static_tools and self._needs_basic_tools(intents):
                # Include basic tools for simple requests
                basic_tool_names = {"web_search", "memory_search", "basic_math"}
                static_tools = [
                    tool
                    for tool in available_static_tools
                    if getattr(tool, "name", "").lower() in basic_tool_names
                ]

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
            dynamic_tool_specs = (
                await self.engineering_agent.generate_dynamic_tool_specification(
                    user_query=user_query,
                    user_id=user_id,
                    intents=intents,
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
        self, intents: List[IntentAnalysis], user_config
    ) -> bool:
        """
        Determine if dynamic tools should be generated based on intent and user configuration.
        Uses only existing IntentAnalysis properties for decision making.
        """
        # Check user configuration
        if (
            user_config
            and user_config.tool
            and not user_config.tool.enable_tool_generation
        ):
            return False

        # Check if any intent explicitly requires custom tools
        for intent in intents:
            # Check if custom tools are explicitly required
            if intent.requires_custom_tools:
                return True

            # Check if high complexity and tool requirement suggest dynamic tools needed
            if (
                intent.requires_tools
                and intent.complexity_level.value in ["COMPLEX", "SPECIALIZED"]
                and intent.tool_complexity_score > 0.7
            ):
                return True

            # Check if domain specificity and computational requirements suggest custom tools
            if (
                intent.domain_specificity > 0.8
                and intent.computational_requirements.value in ["HIGH", "INTENSIVE"]
            ):
                return True

        return False

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
