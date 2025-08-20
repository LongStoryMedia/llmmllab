"""
Integration helpers for using the dynamic tool system with chat completions
"""

import logging
import re
from models import Message, MessageRole, MessageContent, MessageContentType
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


async def analyze_tool_needs(user_message: Message, pipeline: BasePipeline) -> bool:
    """
    Analyze if the request needs a new dynamic tool
    Args:
        user_message: The user's message
    Returns:
        bool: True if a new tool is needed, False otherwise
    """
    analysis_prompt = f"""
Analyze this user request and determine if it requires creating a custom tool/function:

User request: {"\n".join([c.text for c in user_message.content if c.text])}

Consider if the request:
1. Involves complex calculations or data processing
2. Requires specific algorithms or logic
3. Needs custom data transformation
4. Would benefit from a reusable function

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
    return "YES" in response.upper()
