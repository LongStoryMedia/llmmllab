"""
Complete LangChain integration that preserves existing Pydantic models and streaming interface
"""

import asyncio
import json
import logging
from datetime import datetime as dt
from typing import Any, AsyncIterable, Dict, List, Optional

from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.messages import AIMessageChunk

from models.chat_req import ChatReq
from models.chat_response import ChatResponse
from models.message import Message
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from models.message_role import MessageRole
from pipelines.base_pipeline import BasePipeline
from server.context.conversation import ConversationContext
from server.utils.chat.message import (
    extract_message_text,
    to_lc_message,
    from_lc_message,
)
from server.config import logger
from server.db import storage
from server.tools.rag_tools import MemoryRetrievalTool, WebSearchTool, SummarizationTool
from server.tools.integration import get_tools
from server.tools.dynamic_tool import DynamicToolRunner
from runner.pipelines.factory import pipeline_factory


async def agent_chat_completion(
    conversation_ctx: ConversationContext,
    background_tasks: Optional[BackgroundTasks] = None,
) -> StreamingResponse:
    """
    Enhanced chat completion using full LangChain integration while preserving
    existing ChatResponse streaming interface.
    """
    user_message = conversation_ctx.current_user_message
    if not user_message:
        raise ValueError("No user message found")

    user_text = extract_message_text(user_message).strip()

    # Determine if we should use agentic workflow
    use_agentic = should_use_agentic_workflow(user_text)

    logger.info(
        f"Processing request - agentic: {use_agentic}, conversation: {conversation_ctx.conversation_id}"
    )

    return StreamingResponse(
        stream_agentic_response(
            conversation_ctx=conversation_ctx,
            use_agentic=use_agentic,
            background_tasks=background_tasks,
        ),
        media_type="text/event-stream",
    )


async def stream_agentic_response(
    conversation_ctx: ConversationContext,
    use_agentic: bool,
    background_tasks: Optional[BackgroundTasks] = None,
) -> AsyncIterable[str]:
    """
    Stream responses using LangChain while outputting ChatResponse objects.
    """
    full_response = ""
    model_name = "unknown"

    try:
        # Get model profile
        profile_id = (
            conversation_ctx.user_config.model_profiles.engineering_profile_id
            if use_agentic
            else conversation_ctx.user_config.model_profiles.primary_profile_id
        )

        model_profile = await storage.get_service(
            storage.model_profile
        ).get_model_profile_by_id(profile_id, conversation_ctx.user_config.user_id)

        if not model_profile:
            raise ValueError("Model profile not found")

        model_name = model_profile.name

        pipeline, _ = pipeline_factory.get_pipeline(model_name)

        if use_agentic:
            # Use LangChain agent with RAG tools
            async for chunk in stream_langchain_agent_response(
                pipeline, conversation_ctx
            ):
                full_response += chunk
                yield create_streaming_chunk(chunk, False)
        else:
            # Use LangChain chain for standard response
            async for chunk in stream_langchain_standard_response(
                pipeline, conversation_ctx
            ):
                full_response += chunk
                yield create_streaming_chunk(chunk, False)

        # Send final done message
        yield create_streaming_chunk("", True)

        # Store response in background
        if background_tasks and full_response.strip():
            assistant_message = Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(type=MessageContentType.TEXT, text=full_response)
                ],
            )
            background_tasks.add_task(conversation_ctx.add_message, assistant_message)

    except Exception as e:
        logger.error(f"Error in LangChain streaming: {e}", exc_info=True)
        error_chunk = create_error_chunk(str(e))
        yield json.dumps(error_chunk.dict()) + "\n"


async def stream_langchain_agent_response(
    pipeline: BasePipeline, conversation_ctx: ConversationContext
) -> AsyncIterable[str]:
    """
    Stream response using LangChain agent with RAG tools integrated.
    """
    user_message = conversation_ctx.current_user_message
    assert user_message
    user_text = extract_message_text(user_message)

    # Create RAG tools
    tools = [
        MemoryRetrievalTool(conversation_ctx),
        WebSearchTool(conversation_ctx),
        SummarizationTool(conversation_ctx),
    ]

    # Add any dynamic tools that were created

    tool_needs = await get_tools(user_message, pipeline)

    if tool_needs.dynamic_tool:

        dynamic_tool = DynamicToolRunner(tool_needs.dynamic_tool)
        tools.append(dynamic_tool)

    # Create agent prompt with enhanced context
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an advanced AI assistant with access to powerful tools for information retrieval and processing.

Available tools:
- memory_retrieval: Search conversation history for relevant context
- web_search: Get current information from the web
- summarization: Summarize long conversations for better context
{dynamic_tool_info}

Use these tools strategically to provide comprehensive, accurate responses. Always explain your reasoning and cite your sources when using retrieved information.

Current conversation context:
{conversation_summary}
""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Create agent
    agent = create_structured_chat_agent(llm=pipeline, tools=tools, prompt=prompt)

    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        return_intermediate_steps=True,
    )

    # Prepare context
    conversation_summary = await conversation_ctx.summary_context.get_current_summary()
    dynamic_tool_info = (
        f"- {tool_needs.dynamic_tool.name}: {tool_needs.dynamic_tool.description}"
        if tool_needs.dynamic_tool
        else ""
    )

    # Convert conversation history
    chat_history = [to_lc_message(msg) for msg in conversation_ctx.messages[:-1]]

    # Stream using astream_events for true token streaming
    thinking_phase = True

    async for event in agent_executor.astream_events(
        {
            "input": user_text,
            "chat_history": chat_history,
            "conversation_summary": conversation_summary,
            "dynamic_tool_info": dynamic_tool_info,
        },
        version="v2",
        include_types=["chat_model", "tool"],
    ):
        event_type = event["event"]

        if event_type == "on_chat_model_start":
            if thinking_phase:
                yield "🤔 Analyzing your request and determining which tools to use...\n\n"
                thinking_phase = False

        elif event_type == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content

        elif event_type == "on_tool_start":
            tool_name = event.get("name", "unknown")
            tool_input = event["data"].get("input", {})
            yield f"\n\n🔧 **Using {tool_name}**\n"
            if isinstance(tool_input, dict):
                for key, value in tool_input.items():
                    yield f"   - {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}\n"

        elif event_type == "on_tool_end":
            tool_output = event["data"].get("output", "")
            yield f"✅ **Tool completed successfully**\n\n"


async def stream_langchain_standard_response(
    pipeline, conversation_ctx: ConversationContext
) -> AsyncIterable[str]:
    """
    Stream response using LangChain chain for standard (non-agentic) responses.
    """
    # Create a simple RAG chain

    # Get conversation summary and recent context
    summary = await conversation_ctx.summary_context.get_current_summary()
    recent_messages = (
        conversation_ctx.messages[-5:] if len(conversation_ctx.messages) > 1 else []
    )

    # Format context
    conversation_context = "\n".join(
        [
            f"{msg.role.value}: {extract_message_text(msg)}"
            for msg in recent_messages[:-1]  # Exclude current message
        ]
    )

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant. Use the following context to inform your response:

Conversation Summary: {summary}

Recent Context:
{conversation_context}

Provide a helpful, accurate response based on this context.""",
            ),
            ("human", "{question}"),
        ]
    )

    # Create chain
    chain = (
        RunnableParallel(
            {
                "summary": lambda x: summary,
                "conversation_context": lambda x: conversation_context,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | pipeline
        | StrOutputParser()
    )

    # Get user input
    user_text = extract_message_text(conversation_ctx.current_user_message)

    # Stream the response
    async for chunk in chain.astream(user_text):
        if chunk:
            yield chunk


def should_use_agentic_workflow(user_text: str) -> bool:
    """
    Determine if we should use agentic workflow based on user input.
    """
    agentic_keywords = [
        "search",
        "find",
        "look up",
        "research",
        "analyze",
        "calculate",
        "compare",
        "what's the latest",
        "current",
        "recent",
        "news",
        "remember",
        "recall",
        "summarize",
        "explain",
        "break down",
    ]

    user_lower = user_text.lower()
    return any(keyword in user_lower for keyword in agentic_keywords)


def create_streaming_chunk(text: str, done: bool = False) -> str:
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

    response = ChatResponse(
        done=done,
        message=message,
        created_at=dt.now(),
        finish_reason="stop" if done else None,
    )

    return response.model_dump_json() + "\n"


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
        created_at=dt.now(),
        finish_reason="error",
    )


# ============================================================================
# Updated Chat Router Integration
# ============================================================================


async def chat_completion_with_langchain(
    chat_request: ChatReq,
    conversation_ctx: ConversationContext,
    background_tasks: BackgroundTasks,
) -> StreamingResponse:
    """
    Updated chat completion endpoint that uses full LangChain integration.
    This replaces the existing chat completion logic in chat.py
    """
    user_message = conversation_ctx.get_current_user_message(chat_request)
    if not user_message:
        raise ValueError("No user message found in request")

    # Store the current user message in context
    conversation_ctx.current_user_message = user_message

    # Add message and get embeddings
    embeddings, message_id = await conversation_ctx.add_message(user_message)

    # The RAG components are now handled by LangChain tools,
    # so we don't need to manually trigger them here

    # Use enhanced completion logic with LangChain
    return await enhanced_chat_completion_logic(
        conversation_ctx=conversation_ctx,
        background_tasks=background_tasks,
    )


# ============================================================================
# LangChain-Compatible Pipeline Wrapper
# ============================================================================


class PipelineToLangChainLLM:
    """
    Wrapper to make your existing pipeline compatible with LangChain.
    This allows seamless integration without changing your pipeline architecture.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.streaming = True

    async def astream(self, messages, **kwargs):
        """
        Stream tokens from the pipeline in LangChain-compatible format.
        """
        # Convert LangChain messages to your ChatReq format
        chat_req = self._convert_to_chat_req(messages, **kwargs)

        # Use your existing pipeline streaming
        stream = self.pipeline.run(chat_req)

        for chunk in stream:
            if chunk.message and chunk.message.content:
                text_chunk = "".join(
                    content_item.text or ""
                    for content_item in chunk.message.content
                    if content_item.text
                )
                if text_chunk:
                    # Yield in LangChain-compatible format

                    yield AIMessageChunk(content=text_chunk)

    def _convert_to_chat_req(self, messages, **kwargs) -> ChatReq:
        """
        Convert LangChain messages to ChatReq format.
        """
        # Convert LangChain messages to your Message format

        converted_messages = []
        if isinstance(messages, str):
            # Single string input
            converted_messages = [
                Message(
                    role=MessageRole.USER,
                    content=[
                        MessageContent(type=MessageContentType.TEXT, text=messages)
                    ],
                )
            ]
        elif isinstance(messages, list):
            # List of LangChain messages
            converted_messages = [from_lc_message(msg) for msg in messages]

        return ChatReq(
            messages=converted_messages,
            stream=True,
            conversation_id=kwargs.get("conversation_id", 0),
            options=kwargs.get("options"),
        )


# ============================================================================
# Usage Example and Integration Points
# ============================================================================


def integrate_with_existing_system():
    """
    Example of how to integrate this with your existing system.

    Replace these calls in your chat.py:

    1. Replace enhanced_chat_completion_logic with:
       enhanced_chat_completion_logic from this module

    2. Update your chat completion endpoint to use:
       chat_completion_with_langchain

    3. Wrap your pipelines with:
       PipelineToLangChainLLM(your_pipeline)

    4. The RAG tools will handle:
       - Memory retrieval (replaces conversation_ctx.memory_context.retrieve_memories)
       - Web search (replaces conversation_ctx.search_context.search)
       - Summarization (replaces conversation_ctx.summary_context.summarize)
    """
    pass
