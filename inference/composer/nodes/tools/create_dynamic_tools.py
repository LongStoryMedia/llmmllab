import json
from typing import List, cast

from langchain.tools import BaseTool

from models import IntentAnalysis, Tool, DynamicTool, ModelProfileType
from composer.graph.state import WorkflowState
from composer.tools.registry import ToolRegistry
from composer.utils.extraction import extract_content_from_langchain_message
from runner import PipelineFactory, run_pipeline
from utils.model_profile import get_model_profile
from utils.message import extract_message_text
from composer.nodes.base_node import BaseNode


class DynamicToolCreationNode(BaseNode):
    """
    Node responsible for creating dynamic tool specifications based on user queries and intent analysis.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        pipeline_factory: PipelineFactory,
    ):
        super().__init__("dynamic_tool_creation")
        self.tool_registry = tool_registry
        self.pipeline_factory = pipeline_factory

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
                    )
                    raw = extract_message_text(res.message) if res.message else ""

                    # Parse response and check if we should skip tool creation
                    try:
                        parsed_response = json.loads(raw)
                        if parsed_response.get("skip"):
                            self.logger.info(
                                "Skipping dynamic tool creation",
                                user_id=state.user_id,
                                reason=parsed_response.get(
                                    "reason", "No reason provided"
                                ),
                            )
                            continue

                        dt = DynamicTool(**parsed_response)
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        self.logger.error(f"Failed to parse dynamic tool response: {e}")
                        continue

                    # Ensure user_id is set for persistence
                    if not dt.user_id:
                        dt.user_id = state.user_id  # type: ignore

                    # Persist the full dynamic tool immediately (strict failure semantics)
                    try:
                        from db import (
                            storage,
                        )  # pylint: disable=import-outside-toplevel

                        tool_svc = storage.get_service(storage.dynamic_tool)
                        await tool_svc.create_tool(dt)

                        self.logger.info(
                            "Dynamic tool persisted successfully",
                            user_id=state.user_id,
                            tool_name=dt.name,
                        )

                    except Exception as pe:
                        self.logger.error(f"Dynamic tool persistence failed: {pe}")
                        continue

                    # Convert to generic Tool (agent only needs invocation metadata)
                    minimized_fields = {
                        "name": dt.name,
                        "description": dt.description,
                        "args_schema": dt.args_schema,
                        "return_direct": dt.return_direct,
                        "tags": dt.tags,
                        "metadata": dt.metadata,
                        "handle_tool_error": dt.handle_tool_error,
                        "handle_validation_error": dt.handle_validation_error,
                        "response_format": dt.response_format,
                    }
                    state.dynamic_tools.append(Tool(**minimized_fields))

                    self.logger.info(
                        "Dynamic tool created, persisted, and registered",
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

        prompt = f"""As a Tool Engineering Specialist, analyze the user's request and determine if a dynamic tool is needed beyond the available static tools.

User Query: {user_query}
Primary Intent: {intent.primary_intent}
Complexity Level: {intent.complexity_level}
Required Capabilities: {[str(cap) for cap in intent.required_capabilities]}

Available Static Tools: {static_tool_names}

CRITICAL ANALYSIS: Before creating any tool, determine if the user's request can be fulfilled using existing tools:
- If web_search is available and the query needs current information, do NOT create a dynamic tool
- If existing tools can handle the request, respond with: {{"skip": true, "reason": "Existing tools sufficient"}}
- Only create a dynamic tool if there's a genuine capability gap

If a dynamic tool is genuinely needed, create a tool specification that:
1. Addresses specific capability gaps not covered by static tools
2. Is tailored to the user's query and intent
3. Has clear input/output schema definitions
4. Uses real, functional implementation (no fake APIs)
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

Generate either a skip response or a structured tool specification in JSON format matching this schema: {json.dumps(DynamicTool.model_json_schema())}. Focus on practical implementation that directly addresses the user's needs."""
        return prompt
