"""
Strongly typed tool integration system for LangGraph workflows.
Removes code duplication and improves type safety.
"""

import logging
import asyncio
import re
import json
from typing import List, AsyncGenerator, Union, Optional, Dict, Any, cast

from langchain_core.tools import BaseTool
from pydantic import ValidationError

from runner import pipeline_factory, Embeddings
from runner.pipeline_factory import PipelinePriority
from utils.hardware_manager import hardware_manager
from server.tools.dynamic_tool import DynamicToolRunner
from server.db import storage
from server.services.context import ConversationContext
from utils.message import extract_message_text
from models import (
    Message,
    MessageRole,
    MessageContent,
    MessageContentType,
    DynamicTool,
    ToolAnalysisResponse,
    ToolGenerationResult,
)
from .rag_tools import WebSearchTool, MemoryRetrievalTool, SummarizationTool

logger = logging.getLogger(__name__)


class IntentAnalyzer:
    """Analyzes user messages to determine tool requirements."""

    # Keywords that suggest need for tools/computation
    TOOL_INDICATORS = frozenset(
        [
            # Calculation keywords
            "calculate",
            "compute",
            "add",
            "subtract",
            "multiply",
            "divide",
            "sum",
            "average",
            "mean",
            "median",
            "standard deviation",
            "percentage",
            "percent",
            "ratio",
            "proportion",
            # Data processing keywords
            "analyze",
            "process",
            "transform",
            "convert",
            "parse",
            "filter",
            "sort",
            "group",
            "aggregate",
            "summarize",
            # Programming/algorithm keywords
            "algorithm",
            "function",
            "code",
            "script",
            "program",
            "logic",
            "formula",
            "equation",
            "solve",
            # Complex task indicators
            "step by step",
            "break down",
            "systematic",
            "methodical",
            "optimize",
            "find the best",
            "compare options",
        ]
    )

    # Mathematical expression patterns
    MATH_PATTERNS = [
        re.compile(r"\d+\s*[+\-*/]\s*\d+"),  # Basic math operations
        re.compile(r"\d+\s*%"),  # Percentages
        re.compile(r"\$\d+"),  # Currency
        re.compile(r"\d+\.\d+"),  # Decimals
    ]

    COMPUTATION_QUESTIONS = frozenset(
        [
            "how many",
            "how much",
            "what is the",
            "calculate the",
            "find the",
            "determine the",
            "compute the",
        ]
    )

    @classmethod
    def should_use_agentic_workflow(cls, user_message: str) -> bool:
        """
        Determine if a user message would benefit from agentic processing with tools.

        Args:
            user_message: The user's message text

        Returns:
            bool: True if agentic workflow should be used
        """
        message_lower = user_message.lower()

        # Check for tool indicator keywords
        if any(indicator in message_lower for indicator in cls.TOOL_INDICATORS):
            return True

        # Check for mathematical patterns
        if any(pattern.search(user_message) for pattern in cls.MATH_PATTERNS):
            return True

        # Check for computation questions
        if any(question in message_lower for question in cls.COMPUTATION_QUESTIONS):
            return True

        return False

    @classmethod
    def extract_parameters_from_message(
        cls, message: str
    ) -> Dict[str, Union[int, float, str]]:
        """
        Extract parameters from a user message for tool execution.

        Args:
            message: User message text

        Returns:
            dict: Extracted parameters
        """
        params: Dict[str, Union[int, float, str]] = {}

        # Look for numbers
        number_pattern = r"(\d+(?:\.\d+)?)"
        numbers = re.findall(number_pattern, message)

        for i, num_str in enumerate(numbers[:2]):  # Limit to first two numbers
            try:
                if "." in num_str:
                    params[f"number_{i+1}"] = float(num_str)
                else:
                    params[f"number_{i+1}"] = int(num_str)
            except ValueError:
                continue

        # Look for operation type
        message_lower = message.lower()
        if any(op in message_lower for op in ["add", "+", "sum", "plus"]):
            params["operation"] = "add"
        elif any(
            op in message_lower for op in ["subtract", "-", "minus", "difference"]
        ):
            params["operation"] = "subtract"
        elif any(op in message_lower for op in ["multiply", "*", "times", "product"]):
            params["operation"] = "multiply"
        elif any(op in message_lower for op in ["divide", "/"]):
            params["operation"] = "divide"

        return params


class StandardToolProvider:
    """Provides standard RAG tools with proper typing."""

    @staticmethod
    def get_standard_tools(conversation_ctx: ConversationContext) -> List[BaseTool]:
        """Get the standard set of RAG tools."""
        tools: List[BaseTool] = [
            MemoryRetrievalTool(conversation_ctx=conversation_ctx),
            WebSearchTool(conversation_ctx=conversation_ctx),
            SummarizationTool(conversation_ctx=conversation_ctx),
        ]
        return tools


class DynamicToolGenerator:
    """Handles dynamic tool generation with proper error handling."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def analyze_tool_need(
        self, user_message_text: str, conversation_ctx: ConversationContext
    ) -> ToolAnalysisResponse:
        """
        Analyze if a dynamic tool is needed for the user request.

        Args:
            user_message_text: The user's message text
            conversation_ctx: Conversation context

        Returns:
            ToolAnalysisResult with analysis details
        """
        mp = await storage.get_service(storage.model_profile).get_model_profile_by_id(
            conversation_ctx.user_config.model_profiles.analysis_profile_id,
            conversation_ctx.user_config.user_id,
        )

        if not mp:
            raise ValueError("Analysis model profile not found")

        # Use NORMAL priority for tool analysis (used occasionally)

        with pipeline_factory.pipeline(mp, str, PipelinePriority.NORMAL) as pipeline:

            analysis_prompt = f"""
You are a tool analysis assistant. If you are enable to answer a question with sufficient confidence using what you already know, respond negatively.
Analyze this user request and determine if it requires creating a custom tool/function:

User request: {user_message_text}

Consider if the request:
1. Involves complex calculations or data processing that can't be done with basic math
2. Requires specific algorithms or logic beyond simple operations  
3. Needs custom data transformation or analysis
4. Would benefit from a specialized, reusable function
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

Respond with only "NO" if existing tools are sufficient.
If a dynamic tool is needed, describe its purpose and functionality in less than 50 words.
"""

            response = await pipeline.prompt(analysis_prompt)

            # Enforce string return type for analysis pipeline
            if not isinstance(response, str):
                raise TypeError(
                    f"Analysis pipeline returned {type(response).__name__} instead of str. "
                    f"Pipeline declared as str type should only return strings."
                )

            response_text = response.strip()
            # Gating rules:
            # - Hard block if response is exactly 'NO' (any case) or starts with 'NO\n'
            # - Block if response is very short (<= 5 words) or contains obvious negations
            # - Allow only when there's a substantive description
            normalized = response_text.lower()
            is_negative = (
                normalized == "no"
                or normalized.startswith("no\n")
                or "do not" in normalized
                or "does not" in normalized
                or "don't" in normalized
                or "not needed" in normalized
                or "no tool" in normalized
            )
            needs_tool = (not is_negative) and (len(response_text.split()) >= 6)

            return ToolAnalysisResponse(
                needs_dynamic_tool=needs_tool,
                description=(
                    response_text.strip() if needs_tool else "No dynamic tool needed"
                ),
                confidence_score=0.8 if needs_tool else 0.2,
                reasoning=f"Analysis pipeline response: {response_text[:100]}...",
            )

    async def generate_tool(
        self,
        description: str,
        user_message_text: str,
        conversation_ctx: ConversationContext,
    ) -> ToolGenerationResult:
        """
        Generate a dynamic tool based on the analysis.

        Args:
            description: Tool description from analysis
            user_message_text: Original user message
            conversation_ctx: Conversation context

        Returns:
            ToolGenerationResult with success status and tool
        """
        try:
            # Search for existing tools first
            embedding_profile = await storage.get_service(
                storage.model_profile
            ).get_model_profile_by_id(
                conversation_ctx.user_config.model_profiles.embedding_profile_id,
                conversation_ctx.user_config.user_id,
            )

            if not embedding_profile:
                raise ValueError("Embedding profile not found")

            # Create message for embedding
            desc_message = Message(
                role=MessageRole.USER,
                content=[
                    MessageContent(type=MessageContentType.TEXT, text=description)
                ],
            )

            # Use HIGH priority for embeddings (used frequently)
            # (Removed duplicate import of PipelinePriority)

            with pipeline_factory.pipeline(
                embedding_profile, Embeddings, PipelinePriority.HIGH
            ) as pipe:
                embedding_result = await pipe.process_messages([desc_message])
                embedding_vector: List[float] = []
                if (
                    isinstance(embedding_result, list)
                    and embedding_result
                    and isinstance(embedding_result[0], list)
                ):
                    embedding_vector = embedding_result[0]

                # Search for existing tools
                existing_tools, _ = await storage.get_service(
                    storage.dynamic_tool
                ).search_tools_by_embedding(embedding_vector)

                if existing_tools:
                    self.logger.info(f"Found {len(existing_tools)} existing tools")
                    # Return the first matching tool
                    return ToolGenerationResult(success=True, tool=existing_tools[0])

                # Generate new tool
                return await self._generate_new_tool(
                    description, user_message_text, conversation_ctx
                )

        except Exception as e:
            self.logger.error(f"Error generating tool: {e}", exc_info=True)
            return ToolGenerationResult(success=False, error_message=str(e))

    async def _generate_new_tool(
        self,
        description: str,
        user_message_text: str,
        conversation_ctx: ConversationContext,
    ) -> ToolGenerationResult:
        """Generate a completely new dynamic tool."""
        engineering_profile = await storage.get_service(
            storage.model_profile
        ).get_model_profile_by_id(
            conversation_ctx.user_config.model_profiles.engineering_profile_id,
            conversation_ctx.user_config.user_id,
        )

        if not engineering_profile:
            raise ValueError("Engineering profile not found")
        # Try tool generation with memory-aware retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use LOW priority for tool generation (used rarely, can be evicted)
                with pipeline_factory.pipeline(
                    engineering_profile, str, PipelinePriority.LOW
                ) as pipe:
                    generation_prompt = f"""Create a custom tool/function for this user request:

User request: {user_message_text}
Tool description: {description}

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

Format your response as valid JSON matching this schema:
{DynamicTool.model_json_schema()}

Make the tool specific to the user's request but generalizable for similar tasks."""

                    # Add timeout to prevent tool generation from blocking
                    tool_response = await asyncio.wait_for(
                        pipe.process_messages(
                            [
                                Message(
                                    role=MessageRole.USER,
                                    content=[
                                        MessageContent(
                                            type=MessageContentType.TEXT,
                                            text=generation_prompt,
                                        )
                                    ],
                                )
                            ],
                            is_tool_generation=True,  # Deterministic flag for tool generation
                        ),
                        timeout=300.0,  # 5 minute timeout for tool generation (increased for complex tools)
                    )

                    if not tool_response:
                        raise ValueError("No response from engineering pipeline")

                    # Extract and parse JSON
                    json_data = self._extract_json_from_response(tool_response)
                    if not json_data:
                        self.logger.error(
                            f"Could not extract valid JSON from response: {tool_response[:500]}..."
                        )
                        raise ValueError("Could not extract valid JSON from response")

                    # Create DynamicTool
                    try:
                        dynamic_tool = DynamicTool(**json_data)
                        return ToolGenerationResult(success=True, tool=dynamic_tool)
                    except ValidationError as e:
                        self.logger.error(f"Tool validation error: {e}")
                        self.logger.error(f"Invalid JSON data: {json_data}")
                        raise ValueError(f"Tool validation failed: {str(e)}") from e

            except Exception as e:
                if (
                    "out of memory" in str(e).lower()
                    or "failed to allocate" in str(e).lower()
                ):
                    self.logger.warning(
                        f"Tool generation failed due to memory (attempt {attempt + 1}): {e}"
                    )
                    if attempt < max_retries - 1:
                        # Use progressively more aggressive memory clearing
                        try:
                            if attempt == 0:
                                # First retry: normal aggressive cleanup
                                pipeline_factory.force_resource_cleanup(_target_free_memory_gb=1.0)
                            else:
                                # Subsequent retries: nuclear cleanup
                                self.logger.warning(f"Using nuclear cleanup for tool generation retry {attempt + 1}")
                                pipeline_factory.force_memory_cleanup(nuclear_fallback=True)
                        except Exception as cleanup_e:
                            self.logger.warning(f"Memory cleanup failed during retry: {cleanup_e}")
                        
                        import gc
                        gc.collect()
                        continue
                    raise ValueError(
                        f"Tool generation failed after {max_retries} attempts due to memory constraints"
                    ) from e
                raise  # Re-raise non-memory errors immediately

        # If we exit the retry loop without returning, treat as failure
        return ToolGenerationResult(
            success=False,
            error_message="Tool generation retries exhausted without success",
        )

    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response with multiple fallback strategies."""
        if not response or not response.strip():
            self.logger.error("Empty response received")
            return None

        try:
            # Strategy 1: Direct parse
            return cast(Dict[str, Any], json.loads(response.strip()))
        except json.JSONDecodeError:
            pass

        try:
            # Strategy 2: Extract from code blocks
            json_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL | re.IGNORECASE
            )
            if json_match:
                return cast(Dict[str, Any], json.loads(json_match.group(1)))
        except json.JSONDecodeError:
            pass

        try:
            # Strategy 3: Find first complete JSON object with better regex
            # Look for balanced braces more carefully
            start_idx = response.find("{")
            if start_idx != -1:
                brace_count = 0
                in_string = False
                escape_next = False
                end_idx = start_idx

                for i, char in enumerate(response[start_idx:], start_idx):
                    if escape_next:
                        escape_next = False
                        continue

                    if char == "\\":
                        escape_next = True
                        continue

                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue

                    if not in_string:
                        if char == "{":
                            brace_count += 1
                        elif char == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break

                if brace_count == 0:  # Found complete JSON
                    json_str = response[start_idx:end_idx]
                    # Clean up common issues
                    json_str = re.sub(r",\s*}", "}", json_str)
                    json_str = re.sub(r",\s*]", "]", json_str)
                    return cast(Dict[str, Any], json.loads(json_str))
        except (json.JSONDecodeError, ValueError):
            pass

        try:
            # Strategy 4: Try to repair truncated JSON by completing it
            start_idx = response.find("{")
            if start_idx != -1:
                # Take everything from the first brace to the end
                json_candidate = response[start_idx:]

                # Try to complete incomplete JSON structures
                if json_candidate.count("{") > json_candidate.count("}"):
                    # Add missing closing braces
                    missing_braces = json_candidate.count("{") - json_candidate.count(
                        "}"
                    )
                    json_candidate += "}" * missing_braces

                # Remove trailing commas and incomplete entries
                json_candidate = re.sub(r",\s*}", "}", json_candidate)
                json_candidate = re.sub(r",\s*]", "]", json_candidate)

                # Try to parse the repaired JSON
                return cast(Dict[str, Any], json.loads(json_candidate))
        except (json.JSONDecodeError, ValueError):
            pass

        self.logger.error(
            f"Could not extract valid JSON from response: {response[:500]}..."
        )
        return None


class ModernToolManager:
    """
    Modern tool management system replacing the old integration approach.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.intent_analyzer = IntentAnalyzer()
        self.tool_provider = StandardToolProvider()
        self.dynamic_generator = DynamicToolGenerator()

    async def get_tools(
        self, conversation_ctx: ConversationContext
    ) -> AsyncGenerator[Union[str, List[BaseTool]], None]:
        """
        Get all available tools for the conversation context.

        Args:
            conversation_ctx: The conversation context

        Yields:
            Status strings during processing, then final list of tools
        """
        user_message = conversation_ctx.current_user_message
        if not user_message:
            yield "No user message found"
            yield []
            return

        yield "Initializing tool analysis..."

        # Get standard tools
        yield "Loading standard RAG tools..."
        tools: List[BaseTool] = self.tool_provider.get_standard_tools(conversation_ctx)

        user_message_text = extract_message_text(user_message)

        # Analyze if dynamic tool is needed
        yield "Analyzing tool requirements..."

        try:
            # Add timeout to prevent blocking
            analysis = await asyncio.wait_for(
                self.dynamic_generator.analyze_tool_need(
                    user_message_text, conversation_ctx
                ),
                timeout=120.0,  # 2 minute timeout
            )

            if analysis.needs_dynamic_tool:
                yield f"Dynamic tool needed: {analysis.description}"

                # Generate or find dynamic tool with timeout
                tool_result = await asyncio.wait_for(
                    self.dynamic_generator.generate_tool(
                        analysis.description, user_message_text, conversation_ctx
                    ),
                    timeout=300.0,  # 5 minute timeout
                )

                if tool_result.success and tool_result.tool:
                    dynamic_tool_runner = DynamicToolRunner(tool_result.tool)
                    tools.append(dynamic_tool_runner)
                    yield f"✅ Added dynamic tool: {tool_result.tool.name}"
                else:
                    yield f"❌ Failed to create dynamic tool: {tool_result.error_message}"
            else:
                yield "No dynamic tools needed for this request"

        except asyncio.TimeoutError:
            self._tool_timeout_occurred = True  # Flag for nuclear cleanup
            self.logger.warning(
                "Tool generation timed out, continuing with standard tools"
            )
            yield "⏱️ Tool generation timed out, using standard tools"
        except Exception as e:
            self.logger.error(f"Error in dynamic tool analysis: {e}")
            yield f"⚠️ Tool analysis failed, using standard tools: {str(e)[:100]}"

        finally:
            # Clean up memory - use nuclear cleanup for tool generation timeouts
            # as they indicate severe memory pressure
            try:
                if hasattr(self, '_tool_timeout_occurred') and self._tool_timeout_occurred:
                    self.logger.warning("Using nuclear cleanup due to tool generation timeout")
                    hardware_manager.nuclear_clear_memory(kill_processes=False)
                else:
                    hardware_manager.clear_memory(aggressive=True)
            except Exception as cleanup_error:
                self.logger.warning(f"Memory cleanup failed: {cleanup_error}")
                # If cleanup fails, try nuclear as last resort
                try:
                    hardware_manager.nuclear_clear_memory(kill_processes=False)
                except Exception as nuclear_error:
                    self.logger.error(f"Nuclear cleanup also failed: {nuclear_error}")

        # Final yield with completed tools
        yield tools


# Global tool manager instance
tool_manager = ModernToolManager()


async def get_tools(
    conversation_ctx: ConversationContext,
) -> AsyncGenerator[Union[str, List[BaseTool]], None]:
    """
    Main entry point for getting tools - delegates to the modern tool manager.
    """
    async for result in tool_manager.get_tools(conversation_ctx):
        yield result
