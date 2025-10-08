"""
Dynamic tool management for composer workflows.
from composer.tools.dynamic.deduplication import AdvancedToolDeduplicator

from composer.monitoring.logging import composer_logger composer.agents.intent_classifier import IntentClassifierAgent
from composer.tools.dynamic.deduplication import AdvancedToolDeduplicator

from composer.monitoring.logging import composer_loggert for composer workflows.
Handles tool generation, analysis, and deduplication.
"""

import asyncio
import json
import logging

from models import (
    DynamicTool,
    ToolAnalysisResponse,
    ToolGenerationResult,
    ModelProfileType,
)
from utils.model_profile import get_model_profile
from utils.message import extract_message_text
from utils.grammar_generator import parse_structured_output

from db import storage

from runner import pipeline_factory
from runner.pipeline_factory import PipelinePriority
from runner.pipelines.run import run_pipeline

from composer.agents.intent_classifier import IntentClassifierAgent
from composer.tools.dynamic.deduplication import AdvancedToolDeduplicator
from composer.monitoring.logging import composer_logger

logger = composer_logger.logger


class DynamicToolManager:
    """Manages dynamic tool generation with intent analysis and deduplication."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.intent_classifier = IntentClassifierAgent()
        # Initialize deduplicator lazily to avoid import issues
        self._deduplicator = None

    @property
    def deduplicator(self):
        """Lazy loading of deduplicator to avoid circular imports."""
        if self._deduplicator is None:
            # Import deduplicator with error handling for missing dependencies
            self._deduplicator = AdvancedToolDeduplicator()
        return self._deduplicator

    async def analyze_tool_need(
        self, user_message_text: str, user_id: str
    ) -> ToolAnalysisResponse:
        """
        Analyze if a dynamic tool is needed for the user request using intent analysis.

        Uses the IntentClassifierAgent to make intelligent decisions about tool creation
        based on complexity, reusability, and computational requirements.

        Args:
            user_message_text: The user's message text to analyze
            user_id: User ID for retrieving configuration from shared data layer
        """
        try:
            # Get analysis model profile from user config
            mp = await get_model_profile(user_id, ModelProfileType.Analysis)

            # Use NORMAL priority for tool analysis (used occasionally)
            with pipeline_factory.pipeline(
                mp, str, PipelinePriority.NORMAL
            ) as pipeline:
                analysis_prompt = f"""
You are a tool analysis assistant. Determine if this user request requires creating a custom tool/function:

User request: {user_message_text}

Consider if the request:
1. Involves complex calculations or data processing beyond basic operations
2. Requires specific algorithms or logic that would be reusable
3. Needs custom data transformation or analysis
4. Would benefit from a specialized, parameterized function
5. Involves domain-specific processing

Examples that NEED dynamic tools:
- "Calculate compound interest over 5 years with varying rates"
- "Analyze this data pattern and find anomalies" 
- "Create a function to convert between multiple units"
- "Process this text according to specific formatting rules"

Examples that DON'T need dynamic tools:
- "What's 2 + 2?" (basic math)
- "Search for current news" (standard search)
- "What did we discuss earlier?" (memory retrieval)

Respond in the following JSON format:
{json.dumps(ToolAnalysisResponse.model_json_schema())}
"""
                chat_response = await run_pipeline(
                    analysis_prompt, pipeline, grammar=ToolAnalysisResponse
                )
                response_text = (
                    extract_message_text(chat_response.message)
                    if chat_response.message
                    else ""
                )

                # Parse the grammar-constrained response into structured object
                return parse_structured_output(response_text, ToolAnalysisResponse)

        except Exception as e:
            self.logger.error(f"Error analyzing tool need: {e}", exc_info=True)
            return ToolAnalysisResponse(
                needs_dynamic_tool=False,
                description="Error in tool analysis",
                confidence_score=0.0,
                reasoning=f"Analysis failed: {str(e)}",
            )

    async def generate_tool(
        self,
        description: str,
        user_message_text: str,
        user_id: str,
    ) -> ToolGenerationResult:
        """
        Generate a dynamic tool with advanced deduplication.

        Args:
            description: Tool description from analysis
            user_message_text: Original user message
            user_id: User ID for retrieving configuration from shared data layer
        """
        try:
            # Create a proposed tool for deduplication check
            proposed_tool = DynamicTool(
                user_id=user_id,
                name=f"tool_for_{description[:30].replace(' ', '_')}",
                description=description,
                code="# Placeholder - will be generated if no duplicates found",
                function_name="placeholder_function",
                parameters={},
            )

            # Check for duplicates using advanced deduplication
            dedup_result = await self.deduplicator.check_for_duplicates(
                proposed_tool, user_id
            )

            self.logger.info(
                f"Deduplication result: duplicate={dedup_result.is_duplicate}, "
                f"score={dedup_result.similarity_score:.2f}"
            )

            if dedup_result.is_duplicate and dedup_result.existing_tool:
                self.logger.info(
                    f"Found duplicate tool: {dedup_result.existing_tool.name} - "
                    f"{dedup_result.recommendation}"
                )
                return ToolGenerationResult(
                    success=True, tool=dedup_result.existing_tool
                )

            if not dedup_result.should_create_new:
                self.logger.info(
                    f"Deduplication recommends against creation: {dedup_result.recommendation}"
                )
                return ToolGenerationResult(
                    success=False,
                    error_message=f"Tool creation not recommended: {dedup_result.recommendation}",
                )

            # Generate new tool if no duplicates found
            self.logger.info("No duplicates found, generating new tool")
            return await self._generate_new_tool(
                description, user_message_text, user_id
            )

        except Exception as e:
            self.logger.error(f"Error generating tool: {e}", exc_info=True)
            return ToolGenerationResult(success=False, error_message=str(e))

    async def _generate_new_tool(
        self,
        description: str,
        user_message_text: str,
        user_id: str,
    ) -> ToolGenerationResult:
        """Generate a completely new dynamic tool."""
        # Get user config from shared data layer
        user_config = await storage.get_service(storage.user_config).get_user_config(
            user_id
        )
        if not user_config:
            raise ValueError(f"User config not found for user {user_id}")

        engineering_profile = await get_model_profile(
            user_id,
            ModelProfileType.Engineering,
        )

        if not engineering_profile:
            raise ValueError("Engineering profile not found")

        # Try tool generation with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with pipeline_factory.pipeline(
                    engineering_profile,
                    str,
                    PipelinePriority.LOW,
                    user_config.circuit_breaker,
                ) as pipe:
                    generation_prompt = f"""Create a custom tool/function for this user request:

User request: {user_message_text}
Tool description: {description}

You must respond with a valid JSON object that defines a new tool. Do not use any channel formatting or prefixes.

Generate a tool definition with:
1. A clear, descriptive name (snake_case, no spaces)
2. A detailed description of what it does
3. Python code that implements the functionality
4. Clear parameter definitions

Requirements:
- Use snake_case for names
- Include complete working Python code
- No imports unless absolutely necessary
- Handle edge cases
- Return meaningful results

Format your response as ONLY a valid JSON object matching this exact schema:
{json.dumps(DynamicTool.model_json_schema())}

Respond with ONLY the JSON object, no other text or formatting."""

                    # Add timeout to prevent tool generation from blocking
                    response = await asyncio.wait_for(
                        run_pipeline(generation_prompt, pipe, grammar=DynamicTool),
                        timeout=300.0,  # 5 minute timeout for tool generation
                    )
                    tool_response = (
                        extract_message_text(response.message)
                        if response.message
                        else ""
                    )

                    if not tool_response:
                        raise ValueError("No response from engineering pipeline")

                    dynamic_tool = parse_structured_output(tool_response, DynamicTool)

                    # Store the generated tool in the database
                    stored_tool = await storage.get_service(
                        storage.dynamic_tool
                    ).create_tool(dynamic_tool)

                    self.logger.info(
                        f"Successfully stored dynamic tool: {stored_tool.name}"
                    )
                    return ToolGenerationResult(success=True, tool=stored_tool)

            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Tool generation attempt {attempt + 1} timed out, retrying..."
                )
                if attempt == max_retries - 1:
                    return ToolGenerationResult(
                        success=False,
                        error_message="Tool generation timed out after multiple attempts",
                    )
                continue

            except Exception as e:
                self.logger.error(
                    f"Tool generation attempt {attempt + 1} failed: {e}", exc_info=True
                )
                if attempt == max_retries - 1:
                    return ToolGenerationResult(
                        success=False, error_message=f"Tool generation failed: {str(e)}"
                    )
                continue

        return ToolGenerationResult(
            success=False, error_message="Tool generation failed after all retries"
        )
