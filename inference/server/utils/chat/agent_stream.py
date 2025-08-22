"""
Complete LangChain integration that preserves existing Pydantic models and streaming interface
"""

import json
from datetime import datetime as dt
from typing import Any, AsyncIterable, Dict, List, Optional, cast

from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_community.tools import BaseTool
from langchain_core.callbacks.file import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.schema import StandardStreamEvent

from models.message import Message
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from models.message_role import MessageRole

from runner.pipelines.factory import pipeline_factory

from server.config import logger
from server.context.conversation import ConversationContext
from server.db import storage
from server.tools.integration import (
    create_error_chunk,
    create_streaming_chunk,
    get_tools,
)
from server.utils.chat.message import (
    extract_message_text,
    to_lc_message,
)


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler to capture streaming output from LangChain agents."""

    def __init__(self):
        self.tokens = []
        self.current_step = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when a new token is generated."""
        self.tokens.append(token)

    def on_agent_action(self, action, **kwargs) -> None:
        """Called when agent takes an action."""
        self.current_step = f"Using tool: {action.tool}"

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs
    ) -> None:
        """Called when a tool starts running."""
        tool_name = serialized.get("name", "unknown")
        self.current_step = f"Running {tool_name}..."

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes."""
        self.current_step = "Processing results..."


async def agent_chat_completion(
    conversation_ctx: ConversationContext,
    background_tasks: Optional[BackgroundTasks] = None,
) -> AsyncIterable[str]:
    """
    Enhanced chat completion using full LangChain integration while preserving
    existing ChatResponse streaming interface.
    """
    user_message = conversation_ctx.current_user_message
    if not user_message:
        raise ValueError("No user message found")

    logger.info(
        f"Processing request - agentic conversation: {conversation_ctx.conversation_id}"
    )
    full_response = ""

    try:
        # Use LangChain agent with RAG tools
        async for chunk in stream_langchain_agent_response(conversation_ctx):
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
    conversation_ctx: ConversationContext,
) -> AsyncIterable[str]:
    """
    Stream response using LangChain agent with RAG tools integrated.
    """
    user_message = conversation_ctx.current_user_message
    assert user_message
    user_text = extract_message_text(user_message)
    # Add any dynamic tools that were created

    tools: List[BaseTool] = []
    async for tool in get_tools(conversation_ctx):
        if isinstance(tool, str):
            yield create_streaming_chunk(tool, False)
        elif isinstance(tool, BaseTool):
            tools.append(tool)

    if tools:
        yield create_streaming_chunk("🔧 **Using dynamic tools**\n", False)

    mp = await storage.get_service(storage.model_profile).get_model_profile_by_id(
        conversation_ctx.user_config.model_profiles.primary_profile_id,
        conversation_ctx.user_config.user_id,
    )
    assert mp
    pipeline, _ = pipeline_factory.get_pipeline(mp.model_name)

    # Create agent prompt with enhanced context
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                conversation_ctx.create_system_prompt(
                    system_prompt_base=mp.system_prompt,
                    dynamic_tool_info="".join(
                        [f"- {t.name}: {t.description}\n" for t in tools]
                    ),
                ),
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
        max_iterations=50,
        return_intermediate_steps=True,
        callbacks=[StreamingCallbackHandler()],
    )

    # Convert conversation history
    chat_history = [to_lc_message(msg) for msg in conversation_ctx.messages[:-1]]

    # Stream using astream_events for true token streaming
    thinking_phase = True

    async for event in agent_executor.astream_events(
        {
            "input": user_text,
            "chat_history": chat_history,
        },
        version="v2",
        # include_types=["chat_model", "tool"],
    ):
        evt = cast(StandardStreamEvent, event)
        event_type = evt["event"]

        if event_type == "on_chat_model_start":
            if thinking_phase:
                yield create_streaming_chunk(
                    "🤔 Analyzing your request and determining which tools to use...\n\n",
                    False,
                )
                thinking_phase = False

        elif event_type == "on_chat_model_stream":
            chunk = evt["data"]["chunk"] if "chunk" in evt["data"] else None
            if chunk and hasattr(chunk, "content"):
                yield chunk["content"]

        elif event_type == "on_tool_start":
            tool_name = evt["data"].get("name", "unknown")
            tool_input = evt["data"].get("input", {})
            yield create_streaming_chunk(f"\n\n🔧 **Using {tool_name}**\n", False)
            if isinstance(tool_input, dict):
                for key, value in tool_input.items():
                    yield create_streaming_chunk(
                        f"   - {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}\n",
                        False,
                    )

        elif event_type == "on_tool_end":
            tool_output = event["data"].get("output", "")
            yield create_streaming_chunk(
                f"✅ **Tool completed successfully**\n{tool_output}\n\n", False
            )


# async def stream_langchain_standard_response(
#     pipeline, conversation_ctx: ConversationContext
# ) -> AsyncIterable[str]:
#     """
#     Stream response using LangChain chain for standard (non-agentic) responses.
#     """
#     # Create a simple RAG chain

#     # Get conversation summary and recent context
#     summary = await conversation_ctx.summary_context.get_current_summary()
#     recent_messages = (
#         conversation_ctx.messages[-5:] if len(conversation_ctx.messages) > 1 else []
#     )

#     # Format context
#     conversation_context = "\n".join(
#         [
#             f"{msg.role.value}: {extract_message_text(msg)}"
#             for msg in recent_messages[:-1]  # Exclude current message
#         ]
#     )

#     # Create prompt template
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 """You are a helpful AI assistant. Use the following context to inform your response:

# Conversation Summary: {summary}

# Recent Context:
# {conversation_context}

# Provide a helpful, accurate response based on this context.""",
#             ),
#             ("human", "{question}"),
#         ]
#     )

#     # Create chain
#     chain = (
#         RunnableParallel(
#             {
#                 "summary": lambda x: summary,
#                 "conversation_context": lambda x: conversation_context,
#                 "question": RunnablePassthrough(),
#             }
#         )
#         | prompt
#         | pipeline
#         | StrOutputParser()
#     )

#     # Get user input
#     user_text = extract_message_text(conversation_ctx.current_user_message)

#     # Stream the response
#     async for chunk in chain.astream(user_text):
#         if chunk:
#             yield chunk


# def should_use_agentic_workflow(user_text: str) -> bool:
#     """
#     Determine if we should use agentic workflow based on user input.
#     """
#     agentic_keywords = [
#         "search",
#         "find",
#         "look up",
#         "research",
#         "analyze",
#         "calculate",
#         "compare",
#         "what's the latest",
#         "current",
#         "recent",
#         "news",
#         "remember",
#         "recall",
#         "summarize",
#         "explain",
#         "break down",
#     ]

#     user_lower = user_text.lower()
#     return any(keyword in user_lower for keyword in agentic_keywords)
