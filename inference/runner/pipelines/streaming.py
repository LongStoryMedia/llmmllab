"""
Updated streaming.py - Enhanced for both legacy AgentExecutor and modern LangGraph support.
This maintains backward compatibility while adding LangGraph capabilities.
"""

import hashlib
import logging
from typing import Any, Dict, Optional, Type, cast, List, AsyncIterable
import uuid

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables.schema import StandardStreamEvent
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool

from models import (
    ChatResponse,
    ConversationCtx,
    LangGraphState,
    Message,
    MessageRole,
    MessageContent,
    MessageContentType,
)

from .base_pipeline import BasePipelineCore
from .helpers import (
    to_lc_message,
    extract_message_text,
    create_streaming_chunk,
    serialize_to_json,
)


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler to capture streaming output from LangChain agents."""

    def __init__(self):
        self.tokens = []
        self.current_step = ""
        self.logger = logging.getLogger(__name__)

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when a new token is generated."""
        self.logger.debug(f"New token: {token}")
        self.tokens.append(token)

    def on_agent_action(self, action, **kwargs) -> None:
        """Called when agent takes an action."""
        self.logger.debug(f"Agent action: {action}")
        self.current_step = f"Using tool: {action.tool}"

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs
    ) -> None:
        """Called when a tool starts running."""
        tool_name = serialized.get("name", "unknown")
        self.logger.debug(f"Tool start: {tool_name} with input {input_str}")
        self.current_step = f"Running {tool_name}..."

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes."""
        self.logger.debug(f"Tool end with output: {output}")
        self.current_step = "Processing results..."


# on_chat_model_start
# on_llm_start
# on_chat_model_stream
# on_llm_stream
# on_chat_model_end
# on_llm_end
# on_tool_start
# on_tool_end


class EventStreamProcessor:
    """
    Process streaming events from LangChain agents.
    Enhanced with safety mechanisms to prevent infinite loops and repetition.
    """

    def __init__(self, thinking_phase: bool):
        self.thinking_phase = thinking_phase
        # Track the last few tokens to detect repetitions
        self.last_content = ""
        self.repetition_count = 0
        # Set max repetition before terminating
        self.max_repetitions = 3
        # Repetition detection buffer size
        self.repetition_buffer_size = 200
        # Track total generated content length
        self.total_content_length = 0

    def _is_repetitive(self, new_content: str) -> bool:
        """
        Detect repetitive patterns in the generated text.

        Args:
            new_content: The new content to check for repetitions

        Returns:
            bool: True if repetitive pattern detected
        """
        # Update the buffer with new content
        combined = self.last_content + new_content
        if len(combined) > self.repetition_buffer_size:
            combined = combined[-self.repetition_buffer_size :]
        self.last_content = combined

        # Check for repetitive patterns
        # 1. Check for same phrase repeating
        if len(combined) > 50:
            # Look for a phrase of 20-50 chars repeating
            for phrase_len in range(20, min(50, len(combined) // 2)):
                phrase = combined[-phrase_len:]
                if phrase in combined[:-phrase_len]:
                    return True

        # 2. Check for stuttering (same word multiple times)
        words = combined.split()
        if len(words) >= 8:
            last_words = words[-8:]
            unique_words = set(last_words)
            if len(unique_words) <= 3 and len(last_words) >= 6:
                return True

        return False

    def process_event(self, evt: StandardStreamEvent) -> ChatResponse:
        """
        Process a streaming event from the LangChain agent.
        """
        event_type = evt["event"]

        if event_type in ["on_chat_model_start", "on_llm_start", "on_agent_start"]:
            if self.thinking_phase:
                return create_streaming_chunk(
                    "🤔 Analyzing request..\n\n",
                )

        if event_type in [
            "on_chat_model_stream",
            "on_llm_stream",
            "on_agent_stream",
            "on_tool_stream",
        ]:
            chunk = evt["data"]["chunk"] if "chunk" in evt["data"] else None
            if chunk and hasattr(chunk, "content"):
                content = getattr(chunk, "content", "")
                if content:
                    # Check for content length limit
                    self.total_content_length += len(content)

                    # Check for repetitive patterns
                    if self._is_repetitive(content):
                        self.repetition_count += 1
                        # After several repetitions, terminate
                        if self.repetition_count >= self.max_repetitions:
                            return create_streaming_chunk(
                                "\n\n[Terminating repetitive output]", done=True
                            )

                    # Stream the content if we pass checks
                    return create_streaming_chunk(content)

        if event_type in ["on_chat_model_end", "on_llm_end"]:
            return create_streaming_chunk("\n\n[Model finished]", done=True)

        if event_type == "on_tool_start":
            tool_name = evt["data"].get("name", "unknown")
            tool_input = evt["data"].get("input", {})
            tool_txt = ""
            if isinstance(tool_input, dict):
                for key, value in tool_input.items():
                    tool_txt += f"   - {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}\n"
            if tool_txt:
                return create_streaming_chunk(
                    f"\n\n🔧 **Using {tool_name}**\n{tool_txt}"
                )

            return create_streaming_chunk(f"\n\n🔧 **Using {tool_name}**\n")

        if event_type == "on_tool_end":
            tool_output = evt["data"].get("output", "")
            return create_streaming_chunk(
                f"✅ **Tool completed successfully**\n{tool_output}\n\n"
            )

        if event_type == "on_agent_finish":
            # Final response
            output = evt["data"].get("output", "")
            if output:
                return create_streaming_chunk(f"\n\n**Final Answer:**\n{output}\n")

        return create_streaming_chunk(serialize_to_json(evt))


async def stream_pipeline[P: str | List[List[float]] | ChatResponse](
    messages: List[Message],
    pipeline: BasePipelineCore[P],
    tools: Optional[List[BaseTool]] = None,
) -> AsyncIterable[P]:
    """Execute the LangGraph workflow for chat completion."""

    # Convert messages to LangChain format
    lc_messages = [to_lc_message(msg) for msg in messages]

    # Modern LangGraph pipeline
    graph = pipeline.create_graph(tools)  # type: ignore

    latest_message = messages[-1] if messages else None
    assert latest_message is not None, "No messages provided to the pipeline"

    thread_id = hashlib.md5(
        f"{latest_message.conversation_id}-{len(messages)}".encode()
    ).hexdigest()[:16]

    config = RunnableConfig(
        configurable={"thread_id": f"chat-{thread_id}"},
        tags=["chat", "user"],
        run_id=uuid.uuid4(),
        callbacks=[StreamingCallbackHandler()],
    )

    assert latest_message.content, "Latest message has no content"

    # Execute graph with streaming
    user_input = extract_message_text(latest_message)

    initial_state = LangGraphState(
        messages=lc_messages,  # type: ignore
        user_input=user_input,
        error_count=0,
        max_iterations=10,
        current_iteration=0,
    )

    processor = EventStreamProcessor(thinking_phase=True)

    async for event in graph.astream_events(
        initial_state.model_dump(),
        config=config,
        version="v2",
        include_types=[
            "chat_model",
            "tool",
            "llm",
            "agent",
            "chain",
            "retriever",
            "prompt",
        ],
    ):
        res = processor.process_event(cast(StandardStreamEvent, event))
        if pipeline.type == ChatResponse:
            yield cast(P, res)
        elif pipeline.type == str:
            yield cast(P, extract_message_text(res.message) if res.message else "")
        elif pipeline.type == List[List[float]]:
            yield cast(P, res.context if res.context else [[0.0]])


async def run_pipeline[P: str | List[List[float]] | ChatResponse](
    messages: List[Message],
    pipeline: BasePipelineCore[P],
    tools: Optional[List[BaseTool]] = None,
) -> P:
    """
    Get a full response from the pipeline by aggregating streaming chunks.
    """
    chunks: List[str | List[List[float]]] = []
    async for chunk in stream_pipeline(messages, pipeline, tools):
        if pipeline.type == ChatResponse:
            res = cast(ChatResponse, chunk)
            chunks.append(extract_message_text(res.message) if res.message else "")
        elif pipeline.type == str:
            chunks.append(str(chunk))
        elif pipeline.type == List[List[float]]:
            chunks.append(
                cast(List[List[float]], res.context if res.context else [[0.0]])
            )

    if pipeline.type == ChatResponse:
        return cast(
            P,
            ChatResponse(
                done=True,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text="".join(cast(List[str], chunks)),
                        )
                    ],
                ),
            ),
        )
    if pipeline.type == str:
        return cast(P, "".join(cast(List[str], chunks)))
    if pipeline.type == List[List[float]]:
        return cast(P, [c for chunk in chunks for c in chunk])

    raise RuntimeError(f"invalid type {Type[P]}")
