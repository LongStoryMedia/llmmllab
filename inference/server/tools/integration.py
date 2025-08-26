"""
Integration helpers for using the dynamic tool system with chat completions
"""

from datetime import datetime
import logging
import re
import json
from typing import List, AsyncGenerator, Union

from langchain_community.tools import BaseTool

from server.services.hardware_manager import hardware_manager
from runner.pipelines.factory import pipeline_factory
from server.tools.dynamic_tool import DynamicToolRunner
from server.db import storage
from server.tools.rag_tools import WebSearchTool, MemoryRetrievalTool, SummarizationTool
from server.context.conversation import ConversationContext
from server.utils.chat.message import extract_message_text
from models import (
    ChatResponse,
    Message,
    MessageRole,
    MessageContent,
    MessageContentType,
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


async def get_tools(
    conversation_ctx: ConversationContext,
) -> AsyncGenerator[Union[ChatResponse, List[BaseTool]], None]:
    """
    Analyze if the request needs tools and return available tools.
    This function yields status strings during processing and finally yields the list of tools.

    Args:
        conversation_ctx: The conversation context containing memory and search contexts

    Yields:
        Union[str, List[BaseTool]]:
            - Status messages as strings during tool processing
            - The final list of BaseTool instances as the last yield
    """

    user_message = conversation_ctx.current_user_message
    assert user_message, "No user message found in conversation context"

    yield create_streaming_chunk("Initializing tool analysis...", False)

    mp = await storage.get_service(storage.model_profile).get_model_profile_by_id(
        conversation_ctx.user_config.model_profiles.analysis_profile_id,
        conversation_ctx.user_config.user_id,
    )
    assert mp, "Model profile not found"

    yield create_streaming_chunk("Loading analysis pipeline...", False)
    pipeline, _ = pipeline_factory.get_pipeline(mp.model_name)

    yield create_streaming_chunk("Preparing standard tools...", False)
    tools: List[BaseTool] = [
        MemoryRetrievalTool(conversation_ctx=conversation_ctx),
        WebSearchTool(conversation_ctx=conversation_ctx),
        SummarizationTool(conversation_ctx=conversation_ctx),
    ]

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

    yield create_streaming_chunk("Analyzing if dynamic tools are needed...", False)
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

    hardware_manager.clear_memory(aggressive=True)

    needs_dynamic_tool = response.upper() != "NO" and len(response.split()) > 5

    # If a dynamic tool is needed, generate it
    if needs_dynamic_tool:
        yield create_streaming_chunk(
            "Dynamic tool needed, processing request...", False
        )
        description = response.strip()
        embedding_profile = await storage.get_service(
            storage.model_profile
        ).get_model_profile_by_id(
            conversation_ctx.user_config.model_profiles.embedding_profile_id,
            conversation_ctx.user_config.user_id,
        )
        assert embedding_profile, "Embedding profile not found"

        yield create_streaming_chunk("Loading embedding pipeline...", False)
        embedding_pipeline, _ = pipeline_factory.get_pipeline(
            embedding_profile.model_name
        )
        embedding = await embedding_pipeline.emb(description, True, 768)

        yield create_streaming_chunk("Searching for existing similar tools...", False)
        # first see if there are tools that exist which meet the need
        existing_tools, _ = await storage.get_service(
            storage.dynamic_tool
        ).search_tools_by_embedding(embedding[0])

        if existing_tools:
            yield create_streaming_chunk(
                "Found existing tools that match the request...", False
            )
            for et in existing_tools:
                det = DynamicToolRunner(et)
                tools.append(det)
        else:
            yield create_streaming_chunk(
                "No existing tools found, generating a new custom tool...", False
            )
            # Generate the dynamic tool
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
            yield create_streaming_chunk("Loading engineering pipeline...", False)
            engineering_pipeline, _ = pipeline_factory.get_pipeline(
                engineering_profile.model_name
            )

            yield create_streaming_chunk(
                "Generating custom tool implementation...", False
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
                yield create_streaming_chunk("Parsing tool implementation...", False)
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
                    logger.debug(f"Extracted JSON for tool creation: {json_str}")
                    dynamic_tool_data = json.loads(json_str)

                    # Create a DynamicTool Pydantic model
                    dynamic_tool = DynamicTool(**dynamic_tool_data)

                    tools.append(DynamicToolRunner(dynamic_tool))
                    yield create_streaming_chunk(
                        f"Created custom tool: {dynamic_tool.name}", False
                    )
                else:
                    yield create_streaming_chunk(
                        "Failed to extract valid JSON for tool creation", False
                    )
            except Exception as e:
                error_msg = f"Error parsing dynamic tool response: {e}"
                logger.error(error_msg, exc_info=True)
                yield create_streaming_chunk(f"Error: {error_msg}", False)

    # Final yield with the completed tools list
    yield tools


def create_streaming_chunk(
    text: str, done: bool = False, role: MessageRole = MessageRole.ASSISTANT
) -> ChatResponse:
    """
    Create a streaming chunk as a JSON ChatResponse.
    """
    message = None
    if text or not done:
        message = Message(
            role=role,
            content=(
                [MessageContent(type=MessageContentType.TEXT, text=text)]
                if text
                else []
            ),
        )

    return ChatResponse(
        done=done,
        message=message,
        created_at=datetime.now(),
        finish_reason="stop" if done else None,
    )


def create_streaming_string(res: ChatResponse) -> str:
    """
    Create a streaming string representation.
    """
    return res.model_dump_json() + "\n"


def create_error_chunk(error_message: str) -> ChatResponse:
    """
    Create an error chunk as a ChatResponse.
    """
    return ChatResponse(
        done=True,
        message=Message(
            role=MessageRole.OBSERVER,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text=f"I apologize, but I encountered an error: {error_message}",
                )
            ],
        ),
        model="error",
        created_at=datetime.now(),
        finish_reason="error",
    )


def extract_json_from_response(response: str) -> dict | None:
    """
    Extract JSON from LLM response with multiple fallback strategies.
    """
    try:
        # Strategy 1: Direct parse
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    try:
        # Strategy 2: Extract from code blocks
        json_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL | re.IGNORECASE
        )
        if json_match:
            return json.loads(json_match.group(1))
    except json.JSONDecodeError:
        pass

    try:
        # Strategy 3: Find first complete JSON object
        json_match = re.search(r"(\{.*?\})", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)

            # Fix common issues
            # Remove trailing commas
            json_str = re.sub(r",\s*}", "}", json_str)
            json_str = re.sub(r",\s*]", "]", json_str)

            # Fix unescaped newlines in strings
            json_str = json_str.replace("\n    ", "\\n    ")
            json_str = json_str.replace("\n", "\\n")

            return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    logger.error(f"Could not extract valid JSON from response: {response[:200]}...")
    return None
