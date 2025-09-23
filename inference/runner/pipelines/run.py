"""
Enhanced streaming implementation with proper error handling and type safety.
Fixes critical issues with the current streaming architecture.
"""

import hashlib
import uuid
import logging
import re
from typing import Any, Dict, Optional, List, AsyncIterator, cast, Union
from datetime import datetime, timezone

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables.schema import StandardStreamEvent
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage

from models import (
    ChatResponse,
    Message,
    MessageRole,
    MessageContent,
    MessageContentType,
    EventStreamConfig,
)
from utils.langgraph import build_langgraph_state

from utils.message import (
    to_lc_message,
    extract_message_text,
)
from utils.response import create_streaming_chunk

# from utils.serialization import serialize_to_json  # unused

from .base import BasePipelineCore


# Type aliases for better readability
MessageInput = Union[str, Message, List[Union[str, Message]], List[Message], List[str]]


def _normalize_message_input(
    input_data: MessageInput, role: MessageRole = MessageRole.USER
) -> List[Message]:
    """
    Normalize various input types to a List[Message].

    Args:
        input_data: Can be str, Message, List[str | Message]

    Returns:
        List[Message]: Normalized message list
    """
    if isinstance(input_data, str):
        # Single string -> single Message
        return [
            Message(
                role=role,
                content=[MessageContent(type=MessageContentType.TEXT, text=input_data)],
            )
        ]
    elif isinstance(input_data, Message):
        # Single Message -> list with one Message
        return [input_data]
    elif isinstance(input_data, list):
        if not input_data:
            return []

        # Coerce each item in the list to a Message object
        messages = []
        for item in input_data:
            if isinstance(item, str):
                messages.append(
                    Message(
                        role=role,
                        content=[
                            MessageContent(type=MessageContentType.TEXT, text=item)
                        ],
                    )
                )
            elif isinstance(item, Message):
                messages.append(item)
            else:
                # Convert other types to string, then to Message
                messages.append(
                    Message(
                        role=role,
                        content=[
                            MessageContent(type=MessageContentType.TEXT, text=str(item))
                        ],
                    )
                )
        return messages


class StreamingCallbackHandler(BaseCallbackHandler):
    """Enhanced callback handler with better error handling."""

    def __init__(self):
        self.tokens = []
        self.current_step = ""
        self.logger = logging.getLogger(__name__)
        self.errors = []

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when a new token is generated."""
        try:
            self.tokens.append(token)
        except Exception as e:
            self.logger.error(f"Error handling new token: {e}")
            self.errors.append(str(e))

    def on_agent_action(self, action, **kwargs) -> None:
        """Called when agent takes an action."""
        try:
            self.logger.debug(f"Agent action: {action}")
            self.current_step = f"Using tool: {getattr(action, 'tool', 'unknown')}"
        except Exception as e:
            self.logger.error(f"Error handling agent action: {e}")
            self.errors.append(str(e))

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs
    ) -> None:
        """Called when a tool starts running."""
        try:
            tool_name = serialized.get("name", "unknown")
            self.logger.debug(f"Tool start: {tool_name} with input {input_str}")
            self.current_step = f"Running {tool_name}..."
        except Exception as e:
            self.logger.error(f"Error handling tool start: {e}")
            self.errors.append(str(e))

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes."""
        try:
            self.logger.debug(f"Tool end with output: {output}")
            self.current_step = "Processing results..."
        except Exception as e:
            self.logger.error(f"Error handling tool end: {e}")
            self.errors.append(str(e))

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Called when LLM encounters an error."""
        self.logger.error(f"LLM error: {error}")
        self.errors.append(str(error))


class EventStreamProcessor:
    """Stream processor with configurable repetition controls and pipeline-specific post-processing."""

    def __init__(
        self,
        thinking_phase: bool = True,
        config: Optional[EventStreamConfig] = None,
        pipeline: Optional["BasePipelineCore"] = None,
    ):
        self.config = config or EventStreamConfig(thinking_phase_initial=thinking_phase)
        self.thinking_phase = self.config.thinking_phase_initial
        self.total_content_length = 0
        self._buffer = ""
        self.repetition_count = 0
        self._last_hashes: List[int] = []
        self.logger = logging.getLogger(__name__)
        self.pipeline = pipeline

        # Pre-compute event type sets for faster lookup
        self._start_events = frozenset(
            ["on_chat_model_start", "on_llm_start", "on_agent_start"]
        )
        self._stream_events = frozenset(
            ["on_chat_model_stream", "on_llm_stream", "on_agent_stream"]
        )
        self._end_events = frozenset(["on_chat_model_end", "on_llm_end"])

    def set_pipeline(self, pipeline: Optional["BasePipelineCore"]) -> None:
        """Set the pipeline for post-processing and reset its streaming state."""
        self.pipeline = pipeline
        if self.pipeline and hasattr(self.pipeline, "reset_streaming_state"):
            self.pipeline.reset_streaming_state()

    def finalize_pipeline_streaming(self) -> Optional[ChatResponse]:
        """Call the pipeline's finalize_streaming method if available."""
        if self.pipeline and hasattr(self.pipeline, "finalize_streaming"):
            return self.pipeline.finalize_streaming()
        return None

    def _update_buffer(self, new_content: str) -> str:
        self._buffer = (self._buffer + new_content)[
            -self.config.repetition_buffer_chars :
        ]
        return self._buffer

    def _dedup(self, content: str) -> bool:
        """Return True if this content chunk appears to be a duplicate of recent ones."""
        h = hash(content)
        if h in self._last_hashes:
            return True
        self._last_hashes.append(h)
        if len(self._last_hashes) > self.config.dedup_last_hashes:
            self._last_hashes.pop(0)
        return False

    def _is_repetitive(self, new_content: str) -> bool:
        if not new_content:
            return False

        try:
            buf = self._update_buffer(new_content)

            # n-gram repetition near the tail
            words = buf.split()
            if len(words) > self.config.ngram_max:
                tail_window = words[-max(50, self.config.ngram_max * 4) :]
                prior = (
                    words[: -len(tail_window)] if len(words) > len(tail_window) else []
                )
                prior_text = " ".join(prior)
                for n in range(self.config.ngram_min, self.config.ngram_max + 1):
                    if len(tail_window) < n:
                        break
                    ngram = " ".join(tail_window[-n:])
                    # Require n-gram to be meaningful (>= 3 words default)
                    if ngram.strip().count(" ") + 1 < n:
                        continue
                    if prior_text.count(ngram) >= self.config.ngram_repeat_threshold:
                        return True

            # stutter detection (very lax)
            if len(words) >= self.config.stutter_window_words:
                last = words[-self.config.stutter_window_words :]
                unique_ratio = len(set(last)) / max(1, len(last))
                if unique_ratio <= self.config.stutter_unique_ratio_threshold:
                    return True

            return False
        except Exception as e:
            self.logger.error(f"Error checking repetition: {e}")
            return False

    def process_event(self, evt: StandardStreamEvent) -> Optional[ChatResponse]:
        """Process a streaming event with enhanced error handling."""
        try:
            event_type = evt.get("event", "")

            # Use a dispatch table for better performance
            if event_type in self._start_events:
                if self.thinking_phase:
                    self.thinking_phase = False
                    return create_streaming_chunk(
                        "🤔 Analyzing request...\n\n",
                        done=False,
                        role=MessageRole.OBSERVER,
                    )

            elif event_type in self._stream_events:
                return self._process_stream_chunk(evt)

            elif event_type in self._end_events:
                return create_streaming_chunk("", done=True)

            elif event_type == "on_tool_start":
                return self._process_tool_start(evt)

            elif event_type == "on_tool_end":
                return self._process_tool_end(evt)

            elif event_type == "on_agent_finish":
                return self._process_agent_finish(evt)

            # Skip unknown/unhandled events instead of serializing them
            return None

        except Exception as e:
            self.logger.error(f"Error processing event: {e}")
            return create_streaming_chunk(f"[Error processing event: {str(e)[:50]}...]")

    def _process_stream_chunk(self, evt: StandardStreamEvent) -> Optional[ChatResponse]:
        """Process streaming content chunks using pipeline-specific post-processing."""
        try:
            data = evt.get("data", {})
            chunk = data.get("chunk") if isinstance(data, dict) else None

            if not chunk or not hasattr(chunk, "content"):
                return None

            content = getattr(chunk, "content", "")
            if not content:
                return None

            # Check content length limit
            self.total_content_length += len(content)
            if self.total_content_length > self.config.max_content_length:
                return create_streaming_chunk("\n\n[Content limit reached]", done=True)

            # Drop exact duplicates to reduce flicker
            # if self._dedup(content):
            #     return None

            # Check for repetitive patterns
            if self._is_repetitive(content):
                self.repetition_count += 1
                if self.repetition_count >= self.config.max_repetitions:
                    return create_streaming_chunk(
                        "\n\n[Stopping repetitive output]", done=True
                    )

            # Use pipeline-specific post-processing if available
            if self.pipeline and hasattr(self.pipeline, "process_streaming_token"):
                return self.pipeline.process_streaming_token(content)
            else:
                # Fallback to simple streaming chunk
                return create_streaming_chunk(content)

        except Exception as e:
            self.logger.error(f"Error processing stream chunk: {e}")
            return None

    def _process_tool_start(self, evt: StandardStreamEvent) -> Optional[ChatResponse]:
        """Process tool start events."""
        try:
            data = evt.get("data", {})
            tool_name = data.get("name", "unknown")
            tool_input = data.get("input", {})

            tool_txt = ""
            if isinstance(tool_input, dict):
                for key, value in tool_input.items():
                    str_value = str(value)
                    if len(str_value) > 100:
                        str_value = str_value[:100] + "..."
                    tool_txt += f"   - {key}: {str_value}\n"

            if tool_txt:
                return create_streaming_chunk(
                    f"\n\n🔧 **Using {tool_name}**\n{tool_txt}",
                    done=False,
                    role=MessageRole.OBSERVER,
                )
            else:
                return create_streaming_chunk(
                    f"\n\n🔧 **Using {tool_name}**\n",
                    done=False,
                    role=MessageRole.OBSERVER,
                )

        except Exception as e:
            self.logger.error(f"Error processing tool start: {e}")
            return create_streaming_chunk("\n\n🔧 **Using tool**\n")

    def _process_tool_end(self, evt: StandardStreamEvent) -> Optional[ChatResponse]:
        """Process tool end events."""
        try:
            data = evt.get("data", {})
            tool_output = str(data.get("output", ""))

            if len(tool_output) > 500:  # Limit output length
                tool_output = tool_output[:500] + "..."

            return create_streaming_chunk(
                f"✅ **Tool completed**\n{tool_output}\n\n",
                done=False,
                role=MessageRole.OBSERVER,
            )

        except Exception as e:
            self.logger.error(f"Error processing tool end: {e}")
            return create_streaming_chunk("✅ **Tool completed**\n\n")

    def _process_agent_finish(self, evt: StandardStreamEvent) -> Optional[ChatResponse]:
        """Process agent finish events."""
        try:
            data = evt.get("data", {})
            output = str(data.get("output", ""))

            if output:
                return create_streaming_chunk(
                    f"\n\n**Final Answer:**\n{output}\n",
                    done=False,
                    role=MessageRole.OBSERVER,
                )
            else:
                return create_streaming_chunk("", done=True)

        except Exception as e:
            self.logger.error(f"Error processing agent finish: {e}")
            return create_streaming_chunk("", done=True)


async def stream_pipeline(
    messages: MessageInput,
    pipeline: BasePipelineCore,
    tools: Optional[List[BaseTool]] = None,
) -> AsyncIterator[ChatResponse]:
    """
    Execute the LangGraph workflow for chat completion with enhanced error handling.
    Accepts flexible input: str, Message, List[str], or List[Message].
    """
    logger = logging.getLogger(__name__)

    try:
        # Normalize input to List[Message]
        normalized_messages = _normalize_message_input(messages)

        # Validate inputs
        if not normalized_messages:
            yield create_streaming_chunk("No messages provided", done=True)
            return

        # Enforce that streaming requires ChatResponse-capable pipelines
        try:
            if hasattr(
                pipeline, "allows_return_type"
            ) and not pipeline.allows_return_type(ChatResponse):
                yield create_streaming_chunk(
                    "Pipeline does not support streaming ChatResponse chunks.",
                    done=True,
                )
                return
        except Exception as e:
            logger.error(f"Error checking pipeline type capabilities: {e}")
            yield create_streaming_chunk("Invalid pipeline configuration", done=True)
            return

        # Convert messages to LangChain format
        lc_messages: List[BaseMessage] = []
        for msg in normalized_messages:
            try:
                lc_msg = to_lc_message(msg)
                lc_messages.append(lc_msg)
            except Exception as e:
                logger.error(f"Error converting message: {e}")
                continue

        if not lc_messages:
            yield create_streaming_chunk("No valid messages to process", done=True)
            return

        # Create graph
        try:
            graph = pipeline.create_graph(tools)
        except Exception as e:
            logger.error(f"Error creating graph: {e}")
            yield create_streaming_chunk(
                f"Error creating workflow: {str(e)}", done=True
            )
            return

        # Generate thread ID
        latest_message = normalized_messages[-1]
        thread_content = f"{latest_message.conversation_id}-{len(normalized_messages)}"
        thread_id = hashlib.md5(thread_content.encode()).hexdigest()[:16]

        # Create config
        config = RunnableConfig(
            configurable={"thread_id": f"chat-{thread_id}"},
            tags=["chat", "user"],
            run_id=uuid.uuid4(),
            callbacks=[StreamingCallbackHandler()],
        )

        # Extract user input
        user_input = ""
        if latest_message and latest_message.content:
            user_input = extract_message_text(latest_message)

        # Create initial state via builder to decouple from generated model
        initial_state = build_langgraph_state(lc_messages, user_input)

        # Initialize processor
        processor = EventStreamProcessor(thinking_phase=True, pipeline=pipeline)
        # Pipeline-specific streaming with individual post-processing
        pipeline_type = type(pipeline).__name__
        logger.info(f"Using pipeline-specific streaming for {pipeline_type}")

        # Reset pipeline streaming state
        processor.set_pipeline(pipeline)

        # Stream execution
        try:
            logger.info(f"Starting graph streaming for {len(lc_messages)} messages")
            event_count = 0
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
                event_count += 1
                chunk = processor.process_event(cast(StandardStreamEvent, event))
                if chunk:
                    yield chunk

            logger.info(f"Graph streaming completed with {event_count} events")

        except Exception as e:
            logger.error(f"Error during streaming execution: {e}")
            yield create_streaming_chunk(f"Streaming error: {str(e)}", done=True)

        # Call pipeline finalization before final completion
        final_response = processor.finalize_pipeline_streaming()
        if final_response:
            yield final_response

        # Final completion
        yield create_streaming_chunk("", done=True)

    except Exception as e:
        logger.error(f"Pipeline streaming error: {e}", exc_info=True)
        yield create_streaming_chunk(f"Pipeline error: {str(e)}", done=True)


async def run_pipeline(
    messages: MessageInput,
    pipeline: BasePipelineCore,
    tools: Optional[List[BaseTool]] = None,
) -> ChatResponse:
    """
    Get a complete response from the pipeline by aggregating streaming chunks.
    Accepts flexible input: str, Message, List[str], or List[Message].
    """
    logger = logging.getLogger(__name__)

    try:
        chunks: List[str] = []

        async for chunk in stream_pipeline(messages, pipeline, tools):
            if chunk and chunk.message:
                text = extract_message_text(chunk.message)
                if text:
                    chunks.append(text)

        # Combine all chunks
        full_text = "".join(chunks)

        return ChatResponse(
            done=True,
            message=Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=full_text,
                    )
                ],
            ),
            created_at=datetime.now(timezone.utc),
            finish_reason="stop",
        )

    except Exception as e:
        logger.error(f"Error in run_pipeline: {e}")
        return ChatResponse(
            done=True,
            message=Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=f"Error processing request: {str(e)}",
                    )
                ],
            ),
            created_at=datetime.now(timezone.utc),
            finish_reason="error",
        )


async def embed_pipeline(
    messages: MessageInput,
    pipeline: BasePipelineCore,
) -> List[List[float]]:
    """
    Get embeddings from the pipeline for the given messages.
    This provides a normalized interface for embedding operations.
    Accepts flexible input: str, Message, List[str], or List[Message].
    """
    logger = logging.getLogger(__name__)

    try:
        # Normalize input to List[Message]
        normalized_messages = _normalize_message_input(messages)

        # Validate inputs
        if not normalized_messages:
            logger.warning("No messages provided to embed_pipeline")
            return []

        # Validate that this is an embedding pipeline
        try:
            if hasattr(
                pipeline, "allows_return_type"
            ) and not pipeline.allows_return_type(list):
                logger.error("Pipeline does not support embedding return type")
                return []
        except Exception as e:
            logger.warning(f"Could not validate pipeline type for embeddings: {e}")

        # Process messages through the embedding pipeline
        result = await pipeline.process_messages(normalized_messages)

        # Validate result format
        if isinstance(result, list) and all(isinstance(item, list) for item in result):
            return result
        else:
            logger.warning(f"Unexpected embedding result type: {type(result)}")
            return []

    except Exception as e:
        logger.error(f"Error in embed_pipeline: {e}")
        return []


# Type for pipeline chain steps
PipelineStep = tuple[BasePipelineCore, Optional[str]]


async def chain_pipelines(
    initial_input: MessageInput,
    pipeline_steps: List[PipelineStep],
    tools: Optional[List[BaseTool]] = None,
) -> ChatResponse:
    """
    Chain multiple pipeline calls where the output of one becomes input to the next.

    Args:
        initial_input: Starting input (str, Message, List[str], or List[Message])
        pipeline_steps: List of (pipeline, optional_prompt) tuples
        tools: Optional tools for pipelines that support them

    Returns:
        ChatResponse: Final result from the last pipeline

    Example:
        # Chain: user input -> generate -> summarize -> format
        result = await chain_pipelines(
            "What is machine learning?",
            [
                (generation_pipeline, None),  # Generate initial response
                (summary_pipeline, "Summarize this in 2 sentences:"),  # Summarize
                (format_pipeline, "Format as a bullet list:")  # Format
            ]
        )
    """
    logger = logging.getLogger(__name__)

    try:
        if not pipeline_steps:
            raise ValueError("No pipeline steps provided")

        current_input: MessageInput = initial_input

        for i, (pipeline, prompt) in enumerate(pipeline_steps):
            logger.debug(f"Executing pipeline step {i+1}/{len(pipeline_steps)}")

            # Run the pipeline
            result = await run_pipeline(current_input, pipeline, tools)

            # Prepare input for next step
            if i < len(pipeline_steps) - 1:  # Not the last step
                # Extract text from result and optionally combine with prompt
                result_text = ""
                if result.message:
                    result_text = extract_message_text(result.message)

                # For next step, use the result text, optionally with next step's prompt
                next_pipeline, next_prompt = pipeline_steps[i + 1]
                if next_prompt:
                    current_input = f"{next_prompt}\n\n{result_text}"
                else:
                    current_input = result_text
            else:
                # Last step - return the final result
                return result

        # This shouldn't be reached, but just in case
        return ChatResponse(
            done=True,
            message=Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(type=MessageContentType.TEXT, text="Chain completed")
                ],
            ),
            created_at=datetime.now(timezone.utc),
            finish_reason="stop",
        )

    except Exception as e:
        logger.error(f"Error in chain_pipelines: {e}")
        return ChatResponse(
            done=True,
            message=Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=f"Error in pipeline chain: {str(e)}",
                    )
                ],
            ),
            created_at=datetime.now(timezone.utc),
            finish_reason="error",
        )
