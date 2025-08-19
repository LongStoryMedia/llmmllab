"""
Workflow utility functions for handling chat workflows, especially RAG processing.
"""

import re
from typing import Dict, List, Any, Optional

from models.message import Message
from models.message_role import MessageRole
from models.message_content_type import MessageContentType
from models.message_content import MessageContent
from models.model_profile import ModelProfile
from server.context.conversation import ConversationContext


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


def prepare_enhanced_messages(
    conversation_ctx: ConversationContext,
    model_profile: Optional[ModelProfile],
) -> List[Message]:
    """
    Prepare enhanced messages with context from RAG results and conversation summaries.

    Args:
        conversation_ctx: Conversation context
        model_profile: Full model profile

    Returns:
        Enhanced message list with system messages for context
    """
    enhanced_messages = []

    # If we have a model profile with system prompt, use it
    if (
        model_profile
        and hasattr(model_profile, "system_prompt")
        and model_profile.system_prompt
    ):
        enhanced_messages.append(
            Message(
                role=MessageRole.SYSTEM,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=model_profile.system_prompt,
                        url=None,
                    )
                ],
            )
        )

    # Add RAG context as a system message if we have any
    rag_contexts = []

    # Add conversation summaries if available
    if conversation_ctx.summaries:
        summary_texts = [summary.content for summary in conversation_ctx.summaries]
        if summary_texts:
            rag_contexts.append(
                "# Previous Conversation Summaries\n" + "\n\n".join(summary_texts)
            )

    # Also add master summary if available and not already included
    if conversation_ctx.master_summary:
        rag_contexts.append(
            "# Conversation Overview\n" + conversation_ctx.master_summary.content
        )

    # Add memories if available
    if conversation_ctx.retrieved_memories:
        memory_texts = [
            fragment.content
            for memory in conversation_ctx.retrieved_memories
            for fragment in memory.fragments
            if hasattr(fragment, "content")
        ]
        if memory_texts:
            rag_contexts.append("# Relevant Memories\n" + "\n\n".join(memory_texts))

    # Add web search results if available
    if conversation_ctx.search_results:
        search_texts = []
        for result in conversation_ctx.search_results:
            if hasattr(result, "topic") and hasattr(result, "synthesis"):
                search_texts.append(f"## {result.topic}\n{result.synthesis}")
        if search_texts:
            rag_contexts.append("# Web Search Results\n" + "\n\n".join(search_texts))

    # If we have any RAG contexts, add them as a system message
    if rag_contexts:
        enhanced_messages.append(
            Message(
                role=MessageRole.SYSTEM,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text="\n\n".join(rag_contexts),
                        url=None,
                    )
                ],
            )
        )

    # Add all the original messages from the conversation context
    enhanced_messages.extend(conversation_ctx.messages)

    return enhanced_messages


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


def parse_ddg_results(ddg_results: str) -> List[Dict[str, str]]:
    """
    Parse DuckDuckGo search results string into structured data.

    Args:
        ddg_results: String output from DuckDuckGo search

    Returns:
        List of dictionaries with title, link, and snippet
    """
    results = []

    # Simple parsing for the form that DDG usually returns
    try:
        lines = ddg_results.strip().split("\n")
        current_item = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Title:"):
                # If we have a previous item, save it
                if current_item and "title" in current_item:
                    results.append(current_item)
                # Start a new item
                current_item = {"title": line[6:].strip()}
            elif line.startswith("URL:") and current_item:
                current_item["link"] = line[4:].strip()
            elif line.startswith("Description:") and current_item:
                current_item["snippet"] = line[12:].strip()

        # Add the last item if it exists
        if current_item and "title" in current_item:
            results.append(current_item)
    except Exception as e:
        # If parsing fails, return an empty list
        pass

    return results
