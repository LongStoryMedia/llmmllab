"""
Integration helpers for using the dynamic tool system with chat completions
"""

import logging
import re
import json
from typing import Sequence

from langchain_community.tools import BaseTool

from inference.server.tools.dynamic_tool import DynamicToolRunner
from runner.pipelines.factory import pipeline_factory
from server.db import storage
from server.tools.rag_tools import WebSearchTool, MemoryRetrievalTool, SummarizationTool
from server.context.conversation import ConversationContext
from server.utils.chat.message import extract_message_text
from models import (
    Message,
    MessageRole,
    MessageContent,
    MessageContentType,
    ToolNeeds,
    AvailableTool,
    DynamicTool,
)

logger = logging.getLogger(__name__)


def should_use_agentic_workflow(user_message: str) -> bool:
    """
    Determine if a user message would benefit from agentic processing with tools

    Args:
        user_message: The user's message text

    Returns:
        bool: True if agentic workflow should be used
    """
    # Keywords that suggest need for tools/computation
    tool_indicators = [
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

    # Check for mathematical expressions
    math_patterns = [
        r"\d+\s*[+\-*/]\s*\d+",  # Basic math operations
        r"\d+\s*%",  # Percentages
        r"\$\d+",  # Currency
        r"\d+\.\d+",  # Decimals
    ]

    message_lower = user_message.lower()

    # Check for tool indicator keywords
    for indicator in tool_indicators:
        if indicator in message_lower:
            return True

    # Check for mathematical patterns
    for pattern in math_patterns:
        if re.search(pattern, user_message):
            return True

    # Check for question words that might need computation
    computation_questions = [
        "how many",
        "how much",
        "what is the",
        "calculate the",
        "find the",
        "determine the",
        "compute the",
    ]

    for question in computation_questions:
        if question in message_lower:
            return True

    return False


def extract_parameters_from_message(message: str) -> dict:
    """
    Extract parameters from a user message for tool execution

    Args:
        message: User message text

    Returns:
        dict: Extracted parameters
    """
    # This is a simple implementation - a more robust version might use
    # the LLM to extract structured parameters from natural language

    params = {}

    # Look for numbers
    number_pattern = r"(\d+(\.\d+)?)"
    numbers = re.findall(number_pattern, message)
    if numbers:
        for i, (num, _) in enumerate(numbers[:2]):  # Limit to first two numbers
            if "." in num:
                params[f"number_{i+1}"] = float(num)
            else:
                params[f"number_{i+1}"] = int(num)

    # Look for operation type
    if (
        "add" in message.lower()
        or "+" in message
        or "sum" in message.lower()
        or "plus" in message.lower()
    ):
        params["operation"] = "add"
    elif (
        "subtract" in message.lower()
        or "-" in message
        or "minus" in message.lower()
        or "difference" in message.lower()
    ):
        params["operation"] = "subtract"
    elif (
        "multiply" in message.lower()
        or "*" in message
        or "times" in message.lower()
        or "product" in message.lower()
    ):
        params["operation"] = "multiply"
    elif "divide" in message.lower() or "/" in message:
        params["operation"] = "divide"

    return params


async def get_tools(conversation_ctx: ConversationContext) -> Sequence[BaseTool]:
    """
    Analyze if the request needs tools and return available tools
    Args:
        conversation_ctx: The conversation context containing memory and search contexts
    Returns:
        ToolNeeds: A Pydantic model with the following structure:
            - available_tools: List of StaticTool objects (name, description, type)
            - needs_dynamic_tool: Boolean indicating if a dynamic tool is needed
            - dynamic_tool: DynamicTool object if generated, else None
    """

    user_message = conversation_ctx.current_user_message
    assert user_message, "No user message found in conversation context"

    mp = await storage.get_service(storage.model_profile).get_model_profile_by_id(
        conversation_ctx.user_config.model_profiles.analysis_profile_id,
        conversation_ctx.user_config.user_id,
    )
    assert mp, "Model profile not found"

    pipeline, _ = pipeline_factory.get_pipeline(mp.name)

    tools: Sequence[BaseTool] = []

    tools.append(SummarizationTool(conversation_ctx))

    if conversation_ctx.intent.web_search:
        tools.append(WebSearchTool(conversation_ctx))
    if conversation_ctx.intent.memory:
        tools.append(MemoryRetrievalTool(conversation_ctx))

    user_message_text = extract_message_text(user_message)

    # Analyze if a dynamic tool is needed
    analysis_prompt = f"""
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
If a dynamic tool is needed, describe its purpose and functionality in less than 50 words, and only provide the tool definition, without any additional context.
"""

    response = pipeline.get(
        [
            Message(
                role=MessageRole.USER,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=analysis_prompt,
                        url=None,
                    )
                ],
            )
        ],
        mp.parameters,
    )

    needs_dynamic_tool = response.upper() != "NO" and len(response.split()) > 5

    # If a dynamic tool is needed, generate it
    if needs_dynamic_tool:
        description = response.strip()
        embedding_profile = await storage.get_service(
            storage.model_profile
        ).get_model_profile_by_id(
            conversation_ctx.user_config.model_profiles.embedding_profile_id,
            conversation_ctx.user_config.user_id,
        )
        assert embedding_profile, "Embedding profile not found"
        embedding_pipeline, _ = pipeline_factory.get_pipeline(embedding_profile.name)
        embedding = await embedding_pipeline.emb(description, True, 768)
        # first see if there are tools that exist which meet the need
        existing_tools, _ = await storage.get_service(
            storage.dynamic_tool
        ).search_tools_by_embedding(embedding[0])

        if existing_tools:
            et = DynamicToolRunner(existing_tools[0])
            tools.append(et)
        else:
            # Generate the dynamic tool
            generation_prompt = f"""Create a custom tool/function for this user request:

User request: {user_message_text}
Tool description: {description}

Generate a tool definition with:
1. A clear, descriptive name (snake_case, no spaces)
2. A detailed description of what it does
3. Python code that implements the functionality
4. Clear parameter definitions

Format your response as JSON that is valid against this json-schema:
{DynamicTool.model_json_schema()}

Make the tool specific to the user's request but generalizable for similar tasks."""

            engineering_profile = await storage.get_service(
                storage.model_profile
            ).get_model_profile_by_id(
                conversation_ctx.user_config.model_profiles.engineering_profile_id,
                conversation_ctx.user_config.user_id,
            )
            assert engineering_profile, "Engineering profile not found"
            engineering_pipeline, _ = pipeline_factory.get_pipeline(
                engineering_profile.name
            )

            tool_response = engineering_pipeline.get(
                [
                    Message(
                        role=MessageRole.USER,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT,
                                text=generation_prompt,
                                url=None,
                            )
                        ],
                    )
                ],
                engineering_profile.parameters,
            )

            # Parse the JSON response
            try:
                # Extract JSON from the response
                json_match = re.search(
                    r"```json\n(.*?)\n```|```(.*?)```|(\{.*?\})",
                    tool_response,
                    re.DOTALL,
                )
                if json_match:
                    json_str = (
                        json_match.group(1)
                        or json_match.group(2)
                        or json_match.group(3)
                    )
                    dynamic_tool_data = json.loads(json_str)

                    # Create a DynamicTool Pydantic model
                    dynamic_tool = DynamicTool(**dynamic_tool_data)

                    tools.append(DynamicToolRunner(dynamic_tool))
            except Exception as e:
                logger.error(f"Error parsing dynamic tool response: {e}", exc_info=True)

    return tools
