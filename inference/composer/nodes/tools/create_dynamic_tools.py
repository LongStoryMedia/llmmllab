import json
from typing import List, cast

from langchain.tools import BaseTool

from models import IntentAnalysis, Tool, DynamicTool, ModelProfileType
from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger
from composer.tools.registry import ToolRegistry
from composer.utils.extraction import extract_content_from_langchain_message
from runner import PipelineFactory, run_pipeline
from utils.model_profile import get_model_profile
from utils.message import extract_message_text


class DynamicToolCreationNode:
    """
    Node responsible for creating dynamic tool specifications based on user queries and intent analysis.
    """

    def __init__(self, tool_registry: ToolRegistry, pipeline_factory: PipelineFactory):
        self.tool_registry = tool_registry
        self.pipeline_factory = pipeline_factory
        self.logger = composer_logger.logger.bind(component="DynamicToolCreationNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Create dynamic tool specifications based on user query and intent analysis.
        """
        try:
            assert state.user_id
            assert state.intent_classification is not None
            assert state.available_tools is not None
            assert state.current_user_message is not None
            assert state.user_config

            self.logger.info(
                "Creating dynamic tool specification",
                user_id=state.user_id,
            )

            mp = await get_model_profile(state.user_id, ModelProfileType.Engineering)

            with self.pipeline_factory.pipeline(
                mp, str, user_circuit_breaker=state.user_config.circuit_breaker
            ) as pipe:
                for intent in state.intent_classification:
                    # Create prompt for dynamic tool generation
                    prompt = self._create_tool_generation_prompt(
                        user_query=extract_content_from_langchain_message(
                            state.current_user_message
                        ),
                        intent=intent,
                        static_tools=state.available_tools,
                    )
                    res = await run_pipeline(
                        prompt,
                        pipe,
                        cast(List[BaseTool], state.available_tools),
                        DynamicTool,
                    )
                    raw = extract_message_text(res.message) if res.message else ""
                    dt = DynamicTool(**json.loads(raw))
                    
                    # Convert DynamicTool to Tool for WorkflowState compatibility
                    tool_data = dt.model_dump()
                    # Remove DynamicTool-specific fields
                    tool_fields_to_remove = ['id', 'user_id', 'embedding', 'code', 'function_name', 'created_at', 'updated_at']
                    for field in tool_fields_to_remove:
                        tool_data.pop(field, None)
                    
                    t = Tool(**tool_data)
                    state.dynamic_tools.append(t)

                    self.logger.info(
                        "Dynamic tool created and registered",
                        user_id=state.user_id,
                        tool_name=dt.name,
                    )

        except Exception as e:
            self.logger.error(f"Dynamic tool creation failed: {e}")

        return state

    def _create_tool_generation_prompt(
        self, user_query: str, intent: IntentAnalysis, static_tools: List[Tool]
    ) -> str:
        """Create prompt for dynamic tool generation."""

        static_tool_names = [getattr(tool, "name", str(tool)) for tool in static_tools]

        prompt = f"""As a Tool Engineering Specialist, analyze the user's request and generate a dynamic tool specification to address gaps in available capabilities.

User Query: {user_query}
Primary Intent: {intent.primary_intent}
Complexity Level: {intent.complexity_level}
Required Capabilities: {[str(cap) for cap in intent.required_capabilities]}

Available Static Tools: {static_tool_names}

Based on this analysis, create a tool specification that:
1. Addresses specific capability gaps not covered by static tools
2. Is tailored to the user's query and intent
3. Has clear input/output schema definitions
4. Includes proper implementation approach (API calls, calculations, etc.)
5. Considers security and validation requirements

Tool Requirements:
- Must be composable and re-usable
- Should have clear, typed input/output schema
- Should be efficient and focused on single responsibility
- Must not duplicate existing static tool functionality
- Use default error handling (handle_tool_error: false, handle_validation_error: false)

IMPORTANT: Use these exact default values for error handling:
- handle_tool_error: false (not true)
- handle_validation_error: false (not true)
- return_direct: false
- verbose: false

Generate a structured tool specification in JSON format matching this schema: {json.dumps(DynamicTool.model_json_schema())}. Focus on practical implementation that directly addresses the user's needs."""
        return prompt
