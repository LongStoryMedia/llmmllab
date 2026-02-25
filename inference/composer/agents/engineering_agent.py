"""
Engineering Agent for generating technical and engineering responses.
Provides core business logic for technical analysis, code generation, and engineering guidance.
"""

import json
from typing import List, Optional, TYPE_CHECKING, Type

from pydantic import BaseModel

from langchain.tools import BaseTool
from langchain.chat_models import BaseChatModel

from models import (
    ChatResponse,
    Message,
    ModelProfile,
    TechnicalDomain,
    ResponseFormat,
    DynamicTool,
    Tool,
)
from composer.agents.base import BaseAgent
from composer.core.errors import NodeExecutionError

from utils.message_conversion import extract_text_from_message
from utils.logging import llmmllogger

if TYPE_CHECKING:
    from db import DynamicToolStorage


NON_TOOL_NAME = "__NON_TOOL__"


class EngineeringAgent(BaseAgent):
    """
    Engineering Agent for generating technical responses with grammar-constrained output.

    Provides core business logic for technical analysis, code generation, system design,
    and engineering guidance using configured engineering models. Supports tool integration
    and grammar constraints for structured outputs.
    """

    def __init__(
        self,
        model: BaseChatModel,
        profile: ModelProfile,
        tool_storage: "DynamicToolStorage",
    ):
        """
        Initialize engineering agent with required dependencies.

        Args:
            pipeline_factory: Factory for creating engineering pipelines
            profile: Model profile for engineering tasks
            node_metadata: Node execution metadata for tracking
        """
        super().__init__(
            model=model, profile=profile, component_name="EngineeringAgent"
        )
        self.logger = llmmllogger.bind(component="EngineeringAgent")
        self.tool_storage = tool_storage

    async def generate_technical_response(
        self,
        messages: List[Message],
        user_id: str,
        domain: Optional[TechnicalDomain] = TechnicalDomain.GENERAL_ENGINEERING,
        response_format: Optional[ResponseFormat] = ResponseFormat.DETAILED_ANALYSIS,
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[Type[BaseModel]] = None,
    ) -> ChatResponse:
        """
        Generate technical engineering response using configured engineering model.

        Args:
            query: Technical query or problem statement
            user_id: User identifier for model profile retrieval
            domain: Technical domain specialization
            response_format: Desired response format and structure
            tools: Optional tools available to the agent for enhanced capabilities
            grammar: Optional grammar constraints for structured output

        Returns:
            Technical response content
        """
        try:
            self.logger.info(
                "Generating technical response",
                user_id=user_id,
                domain=domain,
                response_format=response_format,
                has_tools=bool(tools),
                has_grammar=bool(grammar),
            )

            # Use BaseAgent's run method with simplified message structure
            return await self.run(
                messages=messages,
                tools=tools,
                grammar=grammar,
            )

        except Exception as e:
            self.logger.error(
                "Technical response generation failed",
                user_id=user_id,
                error=str(e),
            )
            raise NodeExecutionError(
                f"Technical response generation failed: {e}"
            ) from e

    async def generate_dynamic_tool_specification(
        self,
        user_query: str,
        user_id: str,
        static_tools: List[Tool],
    ) -> List[DynamicTool]:
        """
        Generate dynamic tool specification based on user query.

        Args:
            user_query: The user's query/request
            user_id: User identifier
            static_tools: List of available static tools

        Returns:
            A list of DynamicTool specifications.
        """
        dynamic_tools = []
        try:
            self.logger.info(
                "Generating dynamic tool specification",
                query_length=len(user_query),
                has_static_tools=bool(static_tools),
            )

            # Create prompt for dynamic tool generation
            prompt = await self._create_tool_generation_prompt(
                user_query=user_query,
                static_tools=static_tools,
            )

            # Use BaseAgent's run method to get LLM response
            result = await self.run(
                messages=[prompt],
                grammar=DynamicTool,
            )

            # Extract response text
            response_text = (
                extract_text_from_message(result.message)
                if result and result.message
                else ""
            )

            if not response_text.strip():
                self.logger.warning("Empty response from dynamic tool generation")
                return []

            # Parse response and check if we should skip tool creation
            try:
                parsed_response = json.loads(response_text)
                if parsed_response.get("name") == NON_TOOL_NAME:
                    self.logger.info(
                        "Skipping dynamic tool creation",
                        reason="Existing tools sufficient",
                    )
                    return []

                dt = DynamicTool(**parsed_response)

                # Ensure user_id is set for persistence
                if not dt.user_id:
                    dt.user_id = user_id  # type: ignore

                dynamic_tools.append(dt)

                # Persist the dynamic tool
                await self.tool_storage.create_tool(dt)

                self.logger.info(
                    "Dynamic tool specification generated successfully",
                    tool_name=dt.name,
                )
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                self.logger.error(f"Failed to parse dynamic tool response: {e}")

            return dynamic_tools

        except Exception as e:
            self.logger.error(
                "Dynamic tool specification generation failed",
                error=str(e),
            )
            raise NodeExecutionError(
                f"Dynamic tool specification generation failed: {e}"
            ) from e

    async def _create_tool_generation_prompt(
        self,
        user_query: str,
        static_tools: List[Tool],
    ) -> str:
        """Create prompt for dynamic tool generation."""
        static_tool_names = [getattr(tool, "name", str(tool)) for tool in static_tools]
        non_tool = DynamicTool(
            name=NON_TOOL_NAME,
            user_id="",
            code="",
            function_name="",
            description="",
            args_schema={},
            return_direct=False,
            tags=[],
            metadata={},
        )

        prompt = f"""As a Tool Engineering Specialist, analyze the user's request and determine if a dynamic tool is needed beyond the available static tools.

User Query: {user_query}

Available Static Tools: {static_tool_names}

CRITICAL ANALYSIS: Before creating any tool, determine if the user's request can be fulfilled using existing tools:
- If web_search is available and the query needs current information, do NOT create a dynamic tool
- If existing tools can handle the request, respond with: {non_tool.model_dump_json()}
- Only create a dynamic tool if there's a genuine capability gap

If a dynamic tool is genuinely needed, create a tool specification that:
1. Addresses specific capability gaps not covered by static tools
2. Is tailored to the user's query
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
