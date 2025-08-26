"""
Fixed integration helpers for using the dynamic tool system with chat completions
"""

from datetime import datetime
import logging
import re
import json
from typing import List, AsyncGenerator, Union

from langchain_community.tools import BaseTool

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
    """
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
        "percentage",
        "percent",
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
        # Search and retrieval
        "search",
        "find",
        "look up",
        "research",
        "what's the latest",
        "current",
        "recent",
        "news",
        "remember",
        "recall",
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


async def get_tools(
    conversation_ctx: ConversationContext,
) -> AsyncGenerator[Union[ChatResponse, List[BaseTool]], None]:
    """
    Analyze if the request needs tools and return available tools.
    """
    user_message = conversation_ctx.current_user_message
    assert user_message, "No user message found in conversation context"

    yield create_streaming_chunk("Initializing tool analysis...", False)

    # Get model profiles
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

    # Simple check for if we need dynamic tools
    needs_dynamic_tool = should_use_agentic_workflow(user_message_text)

    # If a dynamic tool might be needed, try to generate it
    if needs_dynamic_tool:
        yield create_streaming_chunk("Checking for dynamic tool requirements...", False)

        # First search for existing tools
        embedding_profile = await storage.get_service(
            storage.model_profile
        ).get_model_profile_by_id(
            conversation_ctx.user_config.model_profiles.embedding_profile_id,
            conversation_ctx.user_config.user_id,
        )

        if embedding_profile:
            yield create_streaming_chunk(
                "Searching for existing similar tools...", False
            )
            embedding_pipeline, _ = pipeline_factory.get_pipeline(
                embedding_profile.model_name
            )
            embedding = await embedding_pipeline.emb(user_message_text, True, 768)

            existing_tools, _ = await storage.get_service(
                storage.dynamic_tool
            ).search_tools_by_embedding(embedding[0])

            if existing_tools:
                yield create_streaming_chunk(
                    f"Found {len(existing_tools)} existing tools that match the request...",
                    False,
                )
                for et in existing_tools[:3]:  # Limit to top 3
                    det = DynamicToolRunner(et)
                    tools.append(det)
            else:
                # Try to generate a new tool
                yield create_streaming_chunk("Generating new custom tool...", False)
                dynamic_tool = await generate_dynamic_tool(
                    user_message_text, conversation_ctx
                )
                if dynamic_tool:
                    tools.append(DynamicToolRunner(dynamic_tool))
                    yield create_streaming_chunk(
                        f"Created custom tool: {dynamic_tool.name}", False
                    )

    # Final yield with the completed tools list
    yield tools


async def generate_dynamic_tool(
    user_request: str, conversation_ctx: ConversationContext
) -> DynamicTool | None:
    """
    Generate a dynamic tool for the user request.
    """
    try:
        # Get engineering model profile
        engineering_profile = await storage.get_service(
            storage.model_profile
        ).get_model_profile_by_id(
            conversation_ctx.user_config.model_profiles.engineering_profile_id,
            conversation_ctx.user_config.user_id,
        )
        assert engineering_profile, "Engineering profile not found"

        engineering_pipeline, _ = pipeline_factory.get_pipeline(
            engineering_profile.model_name
        )

        # Improved generation prompt
        generation_prompt = f"""Create a Python function to help with this user request: "{user_request}"

Return ONLY valid JSON with this exact structure:
{{
    "user_id": {conversation_ctx.user_config.user_id},
    "name": "descriptive_function_name",
    "description": "What this tool does",
    "function_name": "function_name_to_call",
    "code": "def function_name_to_call(param1, param2=None):\\n    # Implementation here\\n    return result"
}}

Requirements:
- Use snake_case for names
- Include complete working Python code
- No imports unless absolutely necessary
- Handle edge cases
- Return meaningful results

Example for "calculate compound interest":
{{
    "user_id": {conversation_ctx.user_config.user_id},
    "name": "compound_interest_calculator",
    "description": "Calculates compound interest given principal, rate, time, and compounding frequency",
    "function_name": "calculate_compound_interest",
    "code": "def calculate_compound_interest(principal, annual_rate, years, compounds_per_year=1):\\n    if principal <= 0 or annual_rate < 0 or years < 0 or compounds_per_year <= 0:\\n        return 'Invalid input parameters'\\n    amount = principal * (1 + annual_rate / compounds_per_year) ** (compounds_per_year * years)\\n    interest = amount - principal\\n    return {{  'final_amount': round(amount, 2), 'interest_earned': round(interest, 2) }}"
}}"""

        response = engineering_pipeline.get(
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

        # Better JSON extraction
        json_data = extract_json_from_response(response)
        if not json_data:
            logger.error("Could not extract valid JSON from tool generation response")
            return None

        # Validate required fields
        required_fields = ["user_id", "name", "description", "function_name", "code"]
        if not all(field in json_data for field in required_fields):
            logger.error(f"Missing required fields in tool data: {json_data}")
            return None

        # Create and validate the tool
        dynamic_tool = DynamicTool(**json_data)

        # Basic validation - check if function name appears in code
        if dynamic_tool.function_name not in dynamic_tool.code:
            logger.error("Function name not found in generated code")
            return None

        return dynamic_tool

    except Exception as e:
        logger.error(f"Error generating dynamic tool: {e}", exc_info=True)
        return None


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


def create_streaming_chunk(
    text: str, done: bool = False, role: MessageRole = MessageRole.ASSISTANT
) -> ChatResponse:
    """
    Create a streaming chunk as a JSON ChatResponse.
    """
    message = None
    if text or not done:
        message = Message(
            role=MessageRole.ASSISTANT,
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


def create_streaming_string(res: ChatResponse, done: bool = False) -> str:
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
            role=MessageRole.ASSISTANT,
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
