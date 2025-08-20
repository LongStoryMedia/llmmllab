"""
Simplified chat completion logic that properly streams ChatResponse objects.
"""

import asyncio
import json
from datetime import datetime as dt
from typing import AsyncIterable, List, Optional, Dict, Any

from fastapi import BackgroundTasks, HTTPException, status
from fastapi.responses import StreamingResponse

from models.message import Message
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from models.message_role import MessageRole
from models.chat_response import ChatResponse
from models.chat_req import ChatReq
from server.db import storage
from server.context.conversation import ConversationContext
from server.config import logger
from server.utils.chat.message import extract_message_text
from server.tools.production import ProductionDynamicToolSystem
from runner.pipelines.factory import pipeline_factory


async def enhanced_chat_completion_logic(
    conversation_ctx: ConversationContext,
    background_tasks: Optional[BackgroundTasks] = None,
) -> StreamingResponse:
    """
    Enhanced chat completion logic that replaces the blocking create_agentic_chat_completion.
    This version properly streams responses without blocking.

    Args:
        conversation_ctx: The conversation context containing messages (with RAG data already loaded)
        background_tasks: Optional background tasks for async operations

    Returns:
        StreamingResponse containing the chat completion as ChatResponse JSON objects
    """
    # Get current user message
    user_message = conversation_ctx.current_user_message
    if not user_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No user message found"
        )

    # Extract text for analysis
    user_text = extract_message_text(user_message).strip()

    # Determine if we should use agentic workflow
    # This replaces the should_use_agentic_workflow from integration.py
    use_agentic = should_use_agentic_workflow(user_text)

    logger.info(
        f"Processing request - agentic: {use_agentic}, conversation: {conversation_ctx.conversation_id}"
    )

    # Always stream the response - no more blocking calls
    return StreamingResponse(
        stream_chat_response(
            conversation_ctx=conversation_ctx,
            use_agentic=use_agentic,
            background_tasks=background_tasks,
        ),
        media_type="text/event-stream",
    )


async def stream_chat_response(
    conversation_ctx: ConversationContext,
    use_agentic: bool,
    background_tasks: Optional[BackgroundTasks] = None,
) -> AsyncIterable[str]:
    """
    Stream chat responses as ChatResponse JSON objects.

    Args:
        conversation_ctx: The conversation context
        use_agentic: Whether to use agentic workflow
        background_tasks: Optional background tasks

    Yields:
        JSON strings of ChatResponse objects
    """
    full_response = ""
    model_name = "unknown"

    try:
        # Get the appropriate model profile
        profile_id = (
            conversation_ctx.user_config.model_profiles.engineering_profile_id
            if use_agentic
            else conversation_ctx.user_config.model_profiles.primary_profile_id
        )

        model_profile = await storage.get_service(
            storage.model_profile
        ).get_model_profile_by_id(
            profile_id,
            conversation_ctx.user_config.user_id,
        )

        if not model_profile:
            raise ValueError("Model profile not found")

        model_name = model_profile.name

        # Get pipeline
        pipeline, _ = pipeline_factory.get_pipeline(model_name)

        if use_agentic:
            # Use agentic workflow
            async for chunk in stream_agentic_response(
                pipeline, conversation_ctx, model_profile
            ):
                full_response += chunk
                yield create_streaming_chunk(chunk, model_name, done=False)
        else:
            # Use standard pipeline
            async for chunk in stream_standard_response(
                pipeline, conversation_ctx, model_profile
            ):
                full_response += chunk
                yield create_streaming_chunk(chunk, model_name, done=False)

        # Send final done message
        yield create_streaming_chunk("", model_name, done=True)

        # Store the complete response in background
        if background_tasks and full_response.strip():
            assistant_message = Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT, text=full_response, url=None
                    )
                ],
            )
            background_tasks.add_task(conversation_ctx.add_message, assistant_message)

    except Exception as e:
        logger.error(f"Error in stream_chat_response: {e}", exc_info=True)
        error_chunk = create_error_chunk(str(e))
        yield json.dumps(error_chunk.dict()) + "\n"


async def stream_agentic_response(
    pipeline, conversation_ctx: ConversationContext, model_profile
) -> AsyncIterable[str]:
    """
    Stream response using agentic workflow with production-grade dynamic tool generation.
    Uses the ProductionDynamicToolSystem for robust tool creation, validation, and execution.
    """
    user_message = conversation_ctx.current_user_message
    assert user_message, "User message not found"
    user_text = extract_message_text(user_message)

    # Step 1: Initialize production tool system
    yield "🤖 **Initializing advanced AI tool system...**\n\n"

    # Initialize with marketplace if user ID available
    user_id = (
        conversation_ctx.user_config.user_id if conversation_ctx.user_config else None
    )
    tool_system = ProductionDynamicToolSystem(
        llm=pipeline, enable_marketplace=bool(user_id)
    )
    if user_id and tool_system.marketplace:
        yield "🏪 **Marketplace integration enabled - checking for existing tools...**\n"

    # Step 2: Analyze if we need dynamic tools
    needs_dynamic_tool = await analyze_dynamic_tool_needs(pipeline, user_text)

    dynamic_tool = None
    tool_results = []

    if needs_dynamic_tool:
        yield "🔍 **Analyzing request for tool generation...**\n"

        try:
            # Use production system to create and validate tool
            dynamic_tool = await tool_system.create_and_validate_tool(
                task_description=user_text, user_id=user_id
            )

            if dynamic_tool:
                # Check if this was from marketplace or newly created
                if tool_system.marketplace and user_id:
                    marketplace_tools = tool_system.marketplace.search_tools(user_text)
                    if (
                        marketplace_tools
                        and marketplace_tools[0].name == dynamic_tool.name
                    ):
                        yield f"📦 **Found existing tool in marketplace: {dynamic_tool.name}**\n"
                    else:
                        yield f"⚡ **Generated and validated new tool: {dynamic_tool.name}**\n"
                        yield f"🏷️ **Published to marketplace for future use**\n"
                else:
                    yield f"🛠️ **Generated and validated tool: {dynamic_tool.name}**\n"

                yield f"📝 *{dynamic_tool.description}*\n\n"

                # Execute the tool with monitoring
                yield f"⚙️ **Executing {dynamic_tool.name} with performance monitoring...**\n"

                # Extract parameters from user text for tool execution
                tool_params = extract_tool_parameters(user_text, dynamic_tool)

                # Execute with monitoring
                tool_result = await tool_system.execute_tool_with_monitoring(
                    dynamic_tool, **tool_params
                )

                tool_results.append(
                    f"Dynamic tool '{dynamic_tool.name}' result: {tool_result}"
                )
                yield f"✅ **Tool executed successfully**\n"

                # Show execution stats if available
                if dynamic_tool.name in tool_system.execution_monitor:
                    exec_data = tool_system.execution_monitor[dynamic_tool.name]
                    exec_time = exec_data.get("last_execution_time", 0)
                    yield f"⏱️ *Execution time: {exec_time:.3f}s*\n"

            else:
                yield "⚠️ **Could not generate or validate suitable tool**\n"

        except Exception as e:
            logger.error(f"Error in production tool system: {e}")
            yield f"❌ **Tool system error: {str(e)}**\n"

    # Step 3: Check for standard tools we might also need
    standard_tools_needed = []
    if any(
        word in user_text.lower()
        for word in ["search", "find", "latest", "current", "news"]
    ):
        standard_tools_needed.append("web_search")
    if any(
        word in user_text.lower()
        for word in ["remember", "recall", "previous", "before"]
    ):
        standard_tools_needed.append("memory")

    # Step 4: Execute standard tools
    if standard_tools_needed:
        yield f"🔧 **Using standard RAG tools: {', '.join(standard_tools_needed)}**\n\n"

        # Web search tool
        if "web_search" in standard_tools_needed:
            yield "🔍 *Accessing web search results...*\n"
            if (
                hasattr(conversation_ctx, "search_context")
                and conversation_ctx.search_context
            ):
                tool_results.append("Retrieved current information from web search")
                yield "✓ Current web information integrated\n"

        # Memory tool
        if "memory" in standard_tools_needed:
            yield "🧠 *Accessing conversation memory...*\n"
            if (
                hasattr(conversation_ctx, "memory_context")
                and conversation_ctx.memory_context
            ):
                tool_results.append(
                    "Retrieved relevant memories from conversation history"
                )
                yield "✓ Conversation memories integrated\n"

    yield "\n📊 **Generating comprehensive response with all tool insights...**\n\n"

    # Step 5: Create enhanced prompt with all tool results and RAG context
    tool_context = "\n".join(tool_results) if tool_results else ""

    # Format recent conversation for context
    recent_messages = (
        conversation_ctx.messages[-5:] if len(conversation_ctx.messages) > 1 else []
    )
    conversation_history = format_conversation_for_prompt(recent_messages[:-1])

    # Include comprehensive tool info in prompt
    tool_info = ""
    if dynamic_tool:
        tool_info = f"""
Dynamic Tool Analysis:
- Tool Name: {dynamic_tool.name}
- Description: {dynamic_tool.description}
- Validation: Passed production-grade validation
- Marketplace: {'Available for reuse' if tool_system.marketplace else 'Not applicable'}
"""

    # Get execution statistics for context
    exec_stats = ""
    if hasattr(tool_system, "execution_monitor") and tool_system.execution_monitor:
        stats = tool_system.get_execution_statistics()
        exec_stats = f"Tool System Performance: {stats['success_rate']:.1%} success rate across {stats['total_executions']} executions"

    enhanced_prompt = f"""You are an advanced AI assistant with production-grade dynamic tool generation capabilities. You have analyzed the user's request and created/executed specialized tools as needed.

User's question: {user_text}

Tool Results and Analysis:
{tool_context}

{tool_info}

Recent conversation context:
{conversation_history}

{exec_stats}

Instructions:
1. Provide a comprehensive response using all available tool results
2. If a dynamic tool was created, explain its purpose and how it solved the specific problem
3. Highlight the precision and reliability of the production-grade tool system
4. Reference conversation context and RAG data when relevant
5. Show clear reasoning and methodology
6. If tools were reused from marketplace, mention the efficiency gained

Please provide a detailed, authoritative response:"""

    # Step 6: Stream the enhanced response
    req = ChatReq(
        stream=True,
        messages=[
            Message(
                role=MessageRole.USER,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT, text=enhanced_prompt, url=None
                    )
                ],
            )
        ],
        conversation_id=conversation_ctx.conversation_id,
        options=model_profile.parameters,
    )

    # Stream from pipeline
    try:
        stream = pipeline.run(req)
        for chunk in stream:
            if chunk.message and chunk.message.content:
                text_chunk = "".join(
                    content_item.text or ""
                    for content_item in chunk.message.content
                    if content_item.text
                )
                if text_chunk:
                    yield text_chunk
    except Exception as e:
        logger.error(f"Error in agentic pipeline: {e}")
        yield f"\n\n❌ **Error in response generation:** {str(e)}"


async def stream_agentic_response_simple(
    pipeline, conversation_ctx: ConversationContext, model_profile
) -> AsyncIterable[str]:
    """
    Fallback to simple agentic response if ProductionDynamicToolSystem is not available.
    This is the original simplified implementation.
    """
    user_message = conversation_ctx.current_user_message
    assert user_message, "User message not found"
    user_text = extract_message_text(user_message)

    yield "🔧 **Using simplified tool system (production system not available)...**\n\n"

    # Simple tool detection and execution (original implementation)
    needs_dynamic_tool = await analyze_dynamic_tool_needs(pipeline, user_text)

    dynamic_tool = None
    tool_results = []

    if needs_dynamic_tool:
        yield "🛠️ **Generating basic custom tool...**\n"
        dynamic_tool = await generate_dynamic_tool_streaming(pipeline, user_text)

        if dynamic_tool:
            yield f"✅ **Created tool: {dynamic_tool['name']}**\n"
            try:
                tool_result = await execute_dynamic_tool(dynamic_tool, user_text)
                tool_results.append(f"Tool result: {tool_result}")
                yield f"✓ **Tool executed**\n"
            except Exception as e:
                yield f"❌ Tool execution failed: {str(e)}\n"

    # Continue with standard tools and response generation...
    # (Rest of the original implementation)
    yield "\n📝 **Generating response...**\n\n"

    # Create basic enhanced prompt and stream response
    enhanced_prompt = f"""User asked: {user_text}

Tool results: {'; '.join(tool_results) if tool_results else 'No tools used'}

Please provide a helpful response."""

    req = ChatReq(
        stream=True,
        messages=[
            Message(
                role=MessageRole.USER,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT, text=enhanced_prompt, url=None
                    )
                ],
            )
        ],
        conversation_id=conversation_ctx.conversation_id,
        options=model_profile.parameters,
    )

    try:
        stream = pipeline.run(req)
        for chunk in stream:
            if chunk.message and chunk.message.content:
                text_chunk = "".join(
                    content_item.text or ""
                    for content_item in chunk.message.content
                    if content_item.text
                )
                if text_chunk:
                    yield text_chunk
    except Exception as e:
        logger.error(f"Error in simple agentic pipeline: {e}")
        yield f"\n\n❌ **Error:** {str(e)}"


def extract_tool_parameters(user_text: str, dynamic_tool) -> dict:
    """
    Extract parameters from user text for the dynamic tool.
    This is more sophisticated than the simple version.
    """
    import re

    params = {}

    # Get numbers from text
    numbers = re.findall(r"-?\d+(?:\.\d+)?", user_text)

    # Get tool parameter info if available
    if hasattr(dynamic_tool, "parameters"):
        param_names = dynamic_tool.parameters
    else:
        # Fallback parameter names
        param_names = ["input", "value", "data"]

    # Map numbers to parameters
    for i, param_name in enumerate(param_names):
        if i < len(numbers):
            num_str = numbers[i]
            params[param_name] = float(num_str) if "." in num_str else int(num_str)
        elif param_name in ["input", "data", "text"]:
            # For text parameters, pass the original text
            params[param_name] = user_text

    # If no parameters mapped, provide default
    if not params:
        params["input"] = user_text

    return params


async def analyze_dynamic_tool_needs(pipeline, user_text: str) -> bool:
    """
    Use the LLM to determine if the request needs a dynamic tool.
    """
    analysis_prompt = f"""Analyze this user request and determine if it requires creating a custom tool/function:

User request: {user_text}

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

Respond with only "YES" if a custom dynamic tool would be helpful, "NO" if standard tools are sufficient."""

    try:
        req = ChatReq(
            stream=False,
            messages=[
                Message(
                    role=MessageRole.USER,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT, text=analysis_prompt, url=None
                        )
                    ],
                )
            ],
            conversation_id=0,
        )

        # Use the pipeline to analyze
        response = pipeline.get(req.messages)
        return "YES" in response.upper()
    except Exception as e:
        logger.warning(f"Error analyzing tool needs: {e}")
        return False


async def generate_dynamic_tool_streaming(
    pipeline, user_text: str
) -> Optional[Dict[str, str]]:
    """
    Generate a dynamic tool definition using the LLM.
    """
    generation_prompt = f"""Create a custom tool/function for this user request:

User request: {user_text}

Generate a tool definition with:
1. A clear, descriptive name (snake_case, no spaces)
2. A detailed description of what it does
3. Python code that implements the functionality
4. Clear parameter definitions

Format your response as JSON:
{{
    "name": "tool_name_here",
    "description": "Clear description of what this tool does",
    "code": "def tool_name_here(param1, param2):\\n    # Implementation here\\n    return result",
    "parameters": ["param1", "param2"]
}}

Make the tool specific to the user's request but generalizable for similar tasks."""

    try:
        req = ChatReq(
            stream=False,
            messages=[
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
            conversation_id=0,
        )

        response = pipeline.get(req.messages)

        # Try to extract JSON from response
        import json
        import re

        # Look for JSON in the response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            tool_def = json.loads(json_match.group())
            return tool_def

    except Exception as e:
        logger.error(f"Error generating dynamic tool: {e}")

    return None


async def execute_dynamic_tool(tool_def: Dict[str, str], user_text: str) -> str:
    """
    Execute a dynamically generated tool.
    """
    try:
        # Extract parameters from user text (simplified approach)
        import re

        # Get numbers from the text
        numbers = re.findall(r"-?\d+(?:\.\d+)?", user_text)

        # Create a safe execution environment
        exec_globals = {
            "__builtins__": {
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
            },
            "math": __import__("math"),
            "re": __import__("re"),
        }

        # Execute the tool code
        exec(tool_def["code"], exec_globals)

        # Get the function
        func_name = tool_def["name"]
        tool_func = exec_globals[func_name]

        # Prepare arguments (simplified - you'd want better parameter extraction)
        args = []
        if numbers:
            args = [
                float(n) if "." in n else int(n)
                for n in numbers[: len(tool_def.get("parameters", []))]
            ]

        # Execute the tool
        if args:
            result = tool_func(*args)
        else:
            result = tool_func(user_text)  # Pass the text if no numbers found

        return str(result)

    except Exception as e:
        logger.error(f"Error executing dynamic tool: {e}")
        return f"Tool execution failed: {str(e)}"


async def stream_standard_response(
    pipeline, conversation_ctx: ConversationContext, model_profile
) -> AsyncIterable[str]:
    """
    Stream response using standard pipeline.
    """
    # Create request with all conversation context
    req = ChatReq(
        stream=True,
        messages=conversation_ctx.messages,
        conversation_id=conversation_ctx.conversation_id,
        options=model_profile.parameters,
    )

    # Stream from pipeline
    try:
        stream = pipeline.run(req)
        for chunk in stream:
            if chunk.message and chunk.message.content:
                text_chunk = "".join(
                    content_item.text or ""
                    for content_item in chunk.message.content
                    if content_item.text
                )
                if text_chunk:
                    yield text_chunk
    except Exception as e:
        logger.error(f"Error in standard pipeline: {e}")
        yield f"Error generating response: {str(e)}"


def should_use_agentic_workflow(user_text: str) -> bool:
    """
    Simple heuristic to determine if we should use agentic workflow.
    """
    agentic_keywords = [
        "calculate",
        "compute",
        "analyze",
        "search",
        "find information",
        "look up",
        "research",
        "solve",
        "step by step",
        "break down",
        "what's the latest",
        "current",
        "today",
        "recent",
    ]

    user_lower = user_text.lower()
    return any(keyword in user_lower for keyword in agentic_keywords)


def create_streaming_chunk(text: str, model_name: str, done: bool = False) -> str:
    """
    Create a streaming chunk as a JSON ChatResponse.
    """
    message = None
    if text or not done:
        message = Message(
            role=MessageRole.ASSISTANT,
            content=(
                [MessageContent(type=MessageContentType.TEXT, text=text, url=None)]
                if text
                else []
            ),
        )

    response = ChatResponse(
        done=done,
        message=message,
        model=model_name,
        created_at=dt.now(),
        finish_reason="stop" if done else None,
    )

    return json.dumps(response.dict()) + "\n"


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
                    url=None,
                )
            ],
        ),
        model="error",
        created_at=dt.now(),
        finish_reason="error",
    )


def format_conversation_for_prompt(messages: List[Message]) -> str:
    """
    Format recent messages for inclusion in prompts.
    """
    formatted = []
    for msg in messages:
        text = extract_message_text(msg)
        if text:
            role_name = "User" if msg.role == MessageRole.USER else "Assistant"
            formatted.append(f"{role_name}: {text}")

    return "\n".join(formatted)
