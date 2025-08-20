"""
Integration helpers for using the dynamic tool system with chat completions
"""

import logging
import re
import json

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
from runner.pipelines.base_pipeline import BasePipeline

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


async def analyze_tool_needs(
    user_message: Message, pipeline: BasePipeline
) -> ToolNeeds:
    """
    Analyze if the request needs tools and return available tools
    Args:
        user_message: The user's message
        pipeline: The LLM pipeline to use
    Returns:
        ToolNeeds: A Pydantic model with the following structure:
            - available_tools: List of StaticTool objects (name, description, type)
            - needs_dynamic_tool: Boolean indicating if a dynamic tool is needed
            - dynamic_tool: DynamicTool object if generated, else None
    """
    user_message_text = extract_message_text(user_message)

    # Start with static tools that are always available
    available_tools = [
        AvailableTool(
            name="web_search",
            description="Search the web for information",
            type="static",
        ),
        AvailableTool(
            name="memory_retrieval",
            description="Retrieve information from conversation history",
            type="static",
        ),
    ]

    # Initialize the ToolNeeds model
    result = ToolNeeds(
        available_tools=available_tools, needs_dynamic_tool=False, dynamic_tool=None
    )

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

Available static tools:
- Web search
- Memory retrieval
- Basic conversation

Respond with only "YES" if a custom tool would be helpful, "NO" if existing tools are sufficient.
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
    )

    needs_dynamic_tool = "YES" in response.upper()
    result.needs_dynamic_tool = needs_dynamic_tool

    # If a dynamic tool is needed, generate it
    if needs_dynamic_tool:
        # Generate the dynamic tool
        generation_prompt = f"""Create a custom tool/function for this user request:

User request: {user_message_text}

Generate a tool definition with:
1. A clear, descriptive name (snake_case, no spaces)
2. A detailed description of what it does
3. Python code that implements the functionality
4. Clear parameter definitions

Format your response as JSON that is valid against this json-schema:
{DynamicTool.model_json_schema()}

Make the tool specific to the user's request but generalizable for similar tasks."""

        tool_response = pipeline.get(
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
        )

        # Parse the JSON response
        try:
            # Extract JSON from the response
            json_match = re.search(
                r"```json\n(.*?)\n```|```(.*?)```|(\{.*?\})", tool_response, re.DOTALL
            )
            if json_match:
                json_str = (
                    json_match.group(1) or json_match.group(2) or json_match.group(3)
                )
                dynamic_tool_data = json.loads(json_str)

                # Create a DynamicTool Pydantic model
                dynamic_tool = DynamicTool(**dynamic_tool_data)

                # Set the dynamic tool
                result.dynamic_tool = dynamic_tool

                # Add the dynamic tool to available tools
                result.available_tools.append(
                    AvailableTool(
                        name=dynamic_tool.name,
                        description=dynamic_tool.description,
                        type="dynamic",
                    )
                )
        except Exception as e:
            logger.error(f"Error parsing dynamic tool response: {e}", exc_info=True)

    return result
