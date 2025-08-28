"""
Fixed LangChain integration that resolves prompt template issues
"""

from typing import Any, AsyncIterable, Dict, List, Optional, cast

from fastapi import BackgroundTasks

from langchain.agents import (
    AgentExecutor,
    create_structured_chat_agent,
    create_openai_tools_agent,
)
from langchain_community.tools import BaseTool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.schema import StandardStreamEvent

from langchain_community.chat_models.llamacpp import ChatLlamaCpp
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

from models.message import Message
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from models.message_role import MessageRole
from models.chat_response import ChatResponse
from models.model_profile import ModelProfile

from runner.pipelines.factory import pipeline_factory

from server.config import logger
from server.context.conversation import ConversationContext
from server.db import storage
from server.utils.serialization_utils import serialize_to_json
from server.tools.integration import (
    create_error_chunk,
    create_streaming_chunk,
    create_streaming_string,
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
        f"Processing request - agentic conversation: {conversation_ctx.conversation.id}"
    )
    full_response = ""

    try:
        # Use LangChain agent with RAG tools
        async for chunk in stream_langchain_agent_response(conversation_ctx):
            string_chunk = create_streaming_string(chunk)
            full_response += string_chunk
            yield string_chunk

        # Send final done message
        yield create_streaming_string(ChatResponse(done=True))

        # Store response in background
        if background_tasks and full_response.strip():
            assistant_message = Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(type=MessageContentType.TEXT, text=full_response)
                ],
                conversation_id=conversation_ctx.conversation.id,
            )
            background_tasks.add_task(conversation_ctx.add_message, assistant_message)

    except Exception as e:
        logger.error(f"Error in LangChain streaming: {e}", exc_info=True)
        error_chunk = create_error_chunk(str(e))
        yield serialize_to_json(error_chunk) + "\n"


def create_structured_chat_prompt(
    system_message: str, tools: List[BaseTool]
) -> ChatPromptTemplate:
    """Create a proper structured chat prompt with required variables."""

    # Format tools for the prompt
    tool_strings = []
    tool_names = []

    for tool in tools:
        tool_names.append(tool.name)
        tool_strings.append(f"{tool.name}: {tool.description}")

    tools_text = "\n".join(tool_strings)
    tool_names_text = ", ".join(tool_names)

    # Create the structured chat prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""{system_message}

TOOLS:
------
You have access to the following tools:

{tools_text}

To use a tool, please use the following format:

```
Action: the action to take, should be one of [{tool_names_text}]
Action Input: the input to the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Final Answer: [your response here]
```""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    return prompt


def create_openai_tools_prompt(system_message: str) -> ChatPromptTemplate:
    """Create OpenAI tools-compatible prompt (simpler alternative)."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )


async def stream_langchain_agent_response(
    conversation_ctx: ConversationContext,
) -> AsyncIterable[ChatResponse]:
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

    llm = pipeline_factory.get_pipeline(mp).as_llm()

    system_message = conversation_ctx.create_system_prompt(
        system_prompt_base=mp.system_prompt,
        dynamic_tool_info="".join([f"- {t.name}: {t.description}\n" for t in tools]),
    )

    # Choose agent type based on whether we have tools
    if tools:
        try:
            # Try OpenAI tools agent first (more reliable)
            prompt = create_openai_tools_prompt(system_message)
            agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
        except Exception as e:
            logger.warning(
                f"OpenAI tools agent failed, falling back to structured chat: {e}"
            )
            # Fallback to structured chat agent with proper prompt
            prompt = create_structured_chat_prompt(system_message, tools)
            agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
    else:
        # No tools case - use simple prompt
        prompt = create_openai_tools_prompt(system_message)
        agent = create_openai_tools_agent(llm=llm, tools=[], prompt=prompt)

    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,  # Reduced from 50 to prevent infinite loops
        max_execution_time=120,  # Add timeout
        return_intermediate_steps=True,
        handle_parsing_errors=True,  # Important for error recovery
        callbacks=[StreamingCallbackHandler()],
    )

    # Convert conversation history
    chat_history = [to_lc_message(msg) for msg in conversation_ctx.messages[:-1]]

    # Stream using astream_events for true token streaming
    thinking_phase = True

    try:
        async for event in agent_executor.astream_events(
            {
                "input": user_text,
                "chat_history": chat_history,
            },
            version="v2",
            include_types=["chat_model", "tool", "llm", "agent"],
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
                    # Stream the actual LLM tokens
                    content = getattr(chunk, "content", "")
                    if content:
                        yield create_streaming_chunk(content, False)

            elif event_type == "on_llm_stream":
                chunk = evt["data"]["chunk"] if "chunk" in evt["data"] else None
                if chunk and hasattr(chunk, "content"):
                    # Stream the actual LLM tokens
                    content = getattr(chunk, "content", "")
                    if content:
                        yield create_streaming_chunk(content, False)

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

            elif event_type == "on_agent_finish":
                # Final response
                output = evt["data"].get("output", "")
                if output:
                    yield create_streaming_chunk(
                        f"\n\n**Final Answer:**\n{output}\n", False
                    )

    except Exception as e:
        logger.error(f"Error in agent streaming: {e}", exc_info=True)
        yield create_streaming_chunk(f"❌ Error: {str(e)}\n", False)
