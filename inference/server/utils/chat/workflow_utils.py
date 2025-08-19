"""
Utility functions for determining chat workflow paths.
"""

import re


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


def format_search_query(query: str) -> str:
    """
    Format the user query for better search results.

    Args:
        query: The raw user query

    Returns:
        A formatted query for web search
    """
    # Remove any specific instructions to the AI
    query = re.sub(r"(?i)please\s+", "", query)
    query = re.sub(r"(?i)can you\s+", "", query)
    query = re.sub(r"(?i)I want you to\s+", "", query)
    query = re.sub(r"(?i)I'd like you to\s+", "", query)

    # Remove unnecessary punctuation
    query = re.sub(r"[^\w\s\?\.]", " ", query)

    # Collapse multiple spaces
    query = re.sub(r"\s+", " ", query).strip()

    # Limit length
    if len(query) > 100:
        query = query[:100]

    return query
