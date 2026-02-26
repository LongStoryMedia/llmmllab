"""
Generic workflow execution module for streaming CompiledStateGraph outputs.

This module provides reusable workflow execution capabilities that can be used
across different graph types and state models, extracting the streaming logic
from ComposerService into a generic, reusable component.
"""

import json
import re
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    cast,
)
from datetime import datetime, timezone

from pydantic import BaseModel

from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.schema import StreamEvent, EventData
from langchain_core.messages import AIMessage, ToolMessage

from composer.constants import STRUCTURED_AGENT_RUNNABLE_NAME
from models import (
    IntentAnalysis,
    MessageContentType,
    MessageRole,
    Message,
    MessageContent,
    ChatResponse,
    ModelProfileType,
    Thought,
    ToolCall,
    GenerationState,
)

from runner.pipelines.llamacpp.chat import ReasoningAwareAIMessageChunk
from utils.logging import llmmllogger, serialize_event_data
from db import storage


class WorkflowExecutor:
    """
    Generic workflow executor for CompiledStateGraph streaming.

    Provides reusable streaming execution capabilities that can handle
    any CompiledStateGraph with any state type, as long as the state
    can be converted to a dictionary format.
    """

    def __init__(
        self,
        logger: Optional[Any] = None,
        default_context: str = "workflow_executor",
    ):
        """
        Initialize the workflow executor.

        Args:
            logger: Optional logger instance. If None, uses default llmmllogger
            default_context: Default context string for metadata enrichment
        """
        self.logger = logger or llmmllogger.logger
        self.default_context = default_context

    def create_thread_config(
        self,
        thread_id: str,
        additional_config: Optional[Dict[str, Any]] = None,
    ) -> RunnableConfig:
        """
        Create a thread configuration for workflow checkpointing.

        Args:
            thread_id: Unique thread identifier for checkpointing
            additional_config: Additional configuration parameters

        Returns:
            RunnableConfig: Configuration for LangGraph execution
        """
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        if additional_config:
            config.setdefault("configurable", {}).update(additional_config)

        return config

    # What are some recent advancements in multimodal ai models? what are some of the best open source multimodal models available? of those, which is most capable of interpreting text in an image? Find 4 articles detailing the research and development behind that article and list out the links, publication dates, authors, and a brief synopsis. find any citations within those articles and read through the citations, then list those out as well with their links, publication dates, authors and synopses.

    async def stream_workflow(
        self,
        workflow: CompiledStateGraph,
        initial_state: BaseModel,
        config: Optional[RunnableConfig] = None,
        thread_id: Optional[str] = None,
    ) -> AsyncIterator[ChatResponse]:
        """
        Execute a compiled workflow with streaming output.

        This method provides generic streaming execution for any CompiledStateGraph
        and state type, with optional event enrichment and error handling.

        Args:
            workflow: CompiledStateGraph to execute
            initial_state: Initial state for workflow execution (dict or convertible object)
            config: Optional RunnableConfig. If None and thread_id provided, creates default config
            thread_id: Thread ID for checkpointing (used if config is None)
            enrich_events: Whether to enrich events with metadata and tool_calls
            context_name: Context name for metadata (defaults to default_context)

        Yields:
            Dict[str, Any]: Stream events from workflow execution
        """
        start_time = datetime.now(timezone.utc)
        run_id = ""
        state: Optional[GenerationState] = None
        prev_state: Optional[GenerationState] = state
        analyses_buffer = ""
        contents_buffer = ""
        thoughts_buffer = ""
        raw_think_complete = False
        # Track the run_id associated with the last appended buffers so
        # flushing uses the correct execution id rather than the current
        # event's run_id which can change between events (e.g. tool calls).
        #
        # Reason: streaming chunks for a single model run may be interleaved
        # with other events (tool calls, sub-workflows) that have different
        # `run_id`s. If we flush or key message contents against the
        # current event's `run_id`, we may accidentally associate the
        # buffered text with the wrong execution id, causing the
        # `contents_buffer` to appear empty for the original run. We
        # therefore record the run id at the time we append content.
        last_content_run_id: Optional[str] = None
        last_thoughts_run_id: Optional[str] = None
        last_analyses_run_id: Optional[str] = None
        tool_calls_timer: Dict[str, Dict[str, datetime]] = {}
        tool_calls: Dict[str, ToolCall] = {}
        thoughts: Dict[str, Thought] = {}
        analyses: Dict[str, IntentAnalysis] = {}
        message_contents: Dict[str, MessageContent] = {}
        structured_content: Dict[str, Any] = {}
        total_events = 0

        conversation_id = getattr(initial_state, "conversation_id")
        assert conversation_id is not None and isinstance(
            conversation_id, int
        ), "Initial state must have conversation_id"

        try:
            # Prepare state for execution
            if isinstance(initial_state, dict):
                state_dict = initial_state
            else:
                # Assume state has model_dump method (Pydantic-like)
                if hasattr(initial_state, "model_dump"):
                    state_dict = initial_state.model_dump()
                else:
                    raise ValueError(
                        f"State type {type(initial_state)} must be dict or have model_dump/dict method"
                    )

            # Create config if not provided
            if config is None and thread_id is not None:
                config = self.create_thread_config(thread_id)

            # Stream workflow events
            async for event in workflow.astream_events(
                state_dict,
                config=config,
                version="v2",
            ):
                total_events += 1
                data = event.get("data", {})
                event_type = event.get("event", "unknown")
                chunk = data.get("chunk")
                output = data.get("output", {})
                event_name = event.get("name", "unknown")
                run_id = event.get("run_id", "unknown")
                metadata = event.get("metadata", {})
                new_state = state

                res = ChatResponse(
                    done=False,
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content=[],
                        thoughts=[],
                        tool_calls=[],
                        analyses=[],
                        conversation_id=conversation_id,
                    ),
                    state=state,
                    prev_state=prev_state,
                )

                # make linter happy
                assert res.message
                assert res.message.content is not None
                assert res.message.tool_calls is not None
                assert res.message.analyses is not None
                assert res.message.thoughts is not None

                if event_type == "on_chat_model_start" or event_type == "on_llm_start":
                    md = event.get("metadata", {})
                    task = md.get("task", "Primary")
                    if task == ModelProfileType.Analysis.name:
                        new_state = GenerationState.ANALYZING
                elif event_type == "on_chat_model_end" or event_type == "on_llm_end":
                    if state == GenerationState.ANALYZING:
                        analysis_dict = json.loads(analyses_buffer)
                        analysis = IntentAnalysis(**analysis_dict)
                        analyses[run_id] = analysis
                        res.message.content.append(
                            MessageContent(
                                type=MessageContentType.ANALYSIS, text=analyses_buffer
                            )
                        )
                        res.message.analyses.append(analysis)
                        analyses_buffer = ""
                    # Extract tool calls from model output (proxy/client tool mode)
                    # In proxy mode, tool_calls are in the AIMessage from bind_tools()
                    # but no _tool_start/_tool_end events fire since tools aren't
                    # executed server-side.
                    if isinstance(output, AIMessage):
                        if hasattr(output, "tool_calls") and output.tool_calls:
                            for tc_data in output.tool_calls:
                                tc_id = tc_data.get("id") or run_id
                                tool_call = ToolCall(
                                    name=tc_data.get("name", ""),
                                    args=tc_data.get("args", {}),
                                    execution_id=tc_id,
                                    created_at=datetime.now(timezone.utc),
                                )
                                tool_calls[tc_id] = tool_call
                                res.message.tool_calls.append(tool_call)
                            self.logger.info(
                                "Extracted proxy tool calls from model output",
                                extra={
                                    "tool_count": len(output.tool_calls),
                                    "tool_names": [
                                        tc.get("name") for tc in output.tool_calls
                                    ],
                                },
                            )

                        # Extract text content from non-streaming model output.
                        # When disable_streaming="tool_calling" is active and
                        # tools are bound, no on_chat_model_stream events fire,
                        # so content must be captured here instead.
                        if output.content and last_content_run_id != run_id:
                            text_content, reasoning_content = (
                                self._parse_content_with_reasoning(output.content)
                            )
                            if reasoning_content:
                                for rc_text in reasoning_content:
                                    thoughts_buffer += rc_text
                                    last_thoughts_run_id = run_id
                                    if rc_text.strip():
                                        new_state = GenerationState.THINKING
                                        res.message.thoughts.append(
                                            Thought(text=rc_text)
                                        )
                            if text_content:
                                full_text = "".join(text_content)
                                # Handle <think>...</think> markers that
                                # arrive whole in non-streaming mode.
                                # Split into thoughts (before </think>)
                                # and regular content (after </think>).
                                if "</think>" in full_text:
                                    before, after = full_text.split(
                                        "</think>", 1
                                    )
                                    raw_think_complete = True
                                    # Strip <think> prefix if present
                                    think_text = before.lstrip()
                                    if think_text.startswith("<think>"):
                                        think_text = think_text[
                                            len("<think>") :
                                        ]
                                    if think_text.strip():
                                        thoughts_buffer += think_text
                                        last_thoughts_run_id = run_id
                                        new_state = GenerationState.THINKING
                                        res.message.thoughts.append(
                                            Thought(text=think_text)
                                        )
                                    after = after.strip()
                                    if after:
                                        new_state = GenerationState.RESPONDING
                                        res.message.content.append(
                                            MessageContent(
                                                type=MessageContentType.TEXT,
                                                text=after,
                                            )
                                        )
                                        contents_buffer += after
                                        last_content_run_id = run_id
                                elif full_text.strip():
                                    new_state = GenerationState.RESPONDING
                                    res.message.content.append(
                                        MessageContent(
                                            type=MessageContentType.TEXT,
                                            text=full_text,
                                        )
                                    )
                                    contents_buffer += full_text
                                    last_content_run_id = run_id

                        # Fallback: extract <tool_call> XML from text
                        # content when the model emits them as raw text
                        # (e.g. inside code fences) instead of structured
                        # tool calls that llama.cpp's parser recognises.
                        if (
                            not tool_calls
                            and not (hasattr(output, "tool_calls") and output.tool_calls)
                            and contents_buffer
                            and "<tool_call>" in contents_buffer
                        ):
                            fallback_tcs, cleaned = (
                                self._extract_tool_calls_from_text(
                                    contents_buffer
                                )
                            )
                            if fallback_tcs:
                                self.logger.info(
                                    "Extracted fallback tool calls from text content",
                                    extra={
                                        "tool_count": len(fallback_tcs),
                                        "tool_names": [
                                            tc.name for tc in fallback_tcs
                                        ],
                                    },
                                )
                                for tc in fallback_tcs:
                                    tool_calls[tc.execution_id or run_id] = tc
                                    res.message.tool_calls.append(tc)
                                # The text before the tool calls is
                                # thinking/planning — route it to
                                # thoughts, not content.
                                if cleaned.strip():
                                    thoughts_buffer += cleaned
                                    last_thoughts_run_id = run_id
                                    res.message.thoughts.append(
                                        Thought(text=cleaned.strip())
                                    )
                                # Clear content buffers since the
                                # "content" was really thinking + tool
                                # call XML, not user-visible text.
                                contents_buffer = ""
                                res.message.content.clear()
                                message_contents.clear()
                                last_content_run_id = run_id  # prevent re-processing
                                new_state = GenerationState.EXECUTING
                                # Mark thinking as complete so the final
                                # flush keeps thoughts as thoughts instead
                                # of reclassifying them as content.
                                raw_think_complete = True

                        md = output.response_metadata or {}
                        reason = md.get("finish_reason") or "unknown"
                        token_usage = md.get("token_usage", {})
                        completion_tokens = token_usage.get("completion_tokens", "?")
                        self.logger.debug(
                            "Model generation completed",
                            extra={
                                "finish_reason": reason,
                                "completion_tokens": completion_tokens,
                                "has_tool_calls": bool(
                                    hasattr(output, "tool_calls")
                                    and output.tool_calls
                                ),
                                "content_len": len(contents_buffer),
                            },
                        )
                        if reason == "tool_call" or output.tool_calls:
                            new_state = GenerationState.EXECUTING
                        if reason == "length":
                            self.logger.warn(
                                "Model generation ended due to length",
                                extra={"run_id": run_id},
                            )
                elif (
                    event_type == "on_chat_model_stream"
                    or event_type == "on_llm_stream"
                ) and isinstance(chunk, AIMessage):
                    if state == GenerationState.ANALYZING:
                        for content in self._parse_content(chunk.content):
                            analyses_buffer += content
                            last_analyses_run_id = run_id
                    if hasattr(chunk, "reasoning_content"):
                        reasoning_chunk = cast(ReasoningAwareAIMessageChunk, chunk)
                        rc = reasoning_chunk.reasoning_content
                        thoughts_buffer += rc
                        # Only yield as visible thought if non-whitespace
                        if rc.strip():
                            new_state = GenerationState.THINKING
                            res.message.thoughts.append(Thought(text=rc))
                    elif chunk.content:
                        # Parse content and separate reasoning from regular text
                        text_content, reasoning_content = (
                            self._parse_content_with_reasoning(chunk.content)
                        )

                        # Handle reasoning content
                        if reasoning_content:
                            for reasoning_text in reasoning_content:
                                thoughts_buffer += reasoning_text
                                last_thoughts_run_id = run_id
                                # Only yield as visible thought if non-whitespace
                                if reasoning_text.strip():
                                    new_state = GenerationState.THINKING
                                    res.message.thoughts.append(
                                        Thought(text=reasoning_text)
                                    )

                        # Handle regular text content - detect raw <think> markers
                        if text_content:
                            processed_text: list[str] = []
                            for content_text in text_content:
                                if "</think>" in content_text:
                                    # End of raw think block
                                    raw_think_complete = True
                                    before, after = content_text.split("</think>", 1)
                                    # Everything before </think> is thinking
                                    if before:
                                        thoughts_buffer += before
                                        last_thoughts_run_id = run_id
                                    # Reclassify any accumulated content as thinking
                                    if contents_buffer:
                                        thoughts_buffer = (
                                            contents_buffer + thoughts_buffer
                                        )
                                        contents_buffer = ""
                                        message_contents.clear()
                                    after = after.lstrip("\n")
                                    if after:
                                        processed_text.append(after)
                                elif not raw_think_complete:
                                    # Still in think section - buffer as thoughts,
                                    # do NOT yield as regular content but DO
                                    # yield per-chunk as thoughts so streaming
                                    # consumers can display them.
                                    thoughts_buffer += content_text
                                    last_thoughts_run_id = run_id
                                    # Only yield as visible thought if non-whitespace
                                    if content_text.strip():
                                        new_state = GenerationState.THINKING
                                        res.message.thoughts.append(
                                            Thought(text=content_text)
                                        )
                                else:
                                    processed_text.append(content_text)

                            if processed_text:
                                new_state = GenerationState.RESPONDING
                                for content_text in processed_text:
                                    res.message.content.append(
                                        MessageContent(
                                            type=MessageContentType.TEXT,
                                            text=content_text,
                                        )
                                    )
                                    contents_buffer += content_text
                                    # remember which run this content belongs to so we
                                    # can flush against the correct execution id later
                                    last_content_run_id = run_id
                elif (
                    event_type == "on_chain_end"
                    and event_name == STRUCTURED_AGENT_RUNNABLE_NAME
                ):
                    new_state = GenerationState.FORMATTING
                    if isinstance(output, BaseModel):
                        output = output.model_dump()
                    structured_content = output
                    res.message.structured_output = output

                elif event_type.endswith("_tool_start"):
                    self.logger.info(
                        "Tool call started",
                        extra={"tool_name": event_name, "run_id": run_id},
                    )
                    new_state = GenerationState.EXECUTING
                    tool_calls_timer[run_id] = {"start": datetime.now(timezone.utc)}
                    tool_calls[run_id] = ToolCall(
                        name=event_name,
                        args=data.get("input", {}),
                        execution_id=run_id,
                        created_at=datetime.now(timezone.utc),
                    )
                elif event_type.endswith("_tool_end") and isinstance(
                    output, ToolMessage
                ):
                    end_time = datetime.now(timezone.utc)
                    start_time_tc = tool_calls_timer.get(run_id, {}).get("start")
                    duration_ms = int(
                        (end_time - start_time_tc).total_seconds() * 1000
                        if start_time_tc
                        else 0
                    )

                    tool_call = tool_calls.get(run_id)
                    if tool_call is None:
                        self.logger.warning(
                            "Tool call end event without matching start",
                            extra={"run_id": run_id, "tool_name": event_name},
                        )
                        tool_call = ToolCall(
                            name=event_name,
                            args=data.get("input", {}),
                            execution_id=run_id,
                        )
                    tool_call.success = True
                    tool_call.result_data = output.model_dump()
                    tool_call.execution_time_ms = duration_ms
                    tool_call.execution_id = run_id
                    tool_call.created_at = datetime.now(timezone.utc)
                    tool_calls[run_id] = tool_call
                    self.logger.debug(
                        "Appending tool call to response",
                        extra={"tool_call": str(tool_call)},
                    )
                    res.message.tool_calls.append(tool_call)

                prev_state = state

                if new_state != state:
                    self.logger.debug(f"State transition: {state} -> {new_state}")
                    if state == GenerationState.THINKING:
                        self.logger.debug(
                            f"Thoughts buffer: {thoughts_buffer}\n{'-'*20}"
                        )
                        thoughts[run_id] = Thought(
                            text=thoughts_buffer,
                            created_at=datetime.now(timezone.utc),
                        )
                        thoughts_buffer = ""
                    elif state == GenerationState.ANALYZING:
                        self.logger.debug(
                            f"Analyses buffer: {analyses_buffer}\n{'-'*20}"
                        )
                        try:
                            analysis_dict = json.loads(analyses_buffer)
                            analyses[last_analyses_run_id or run_id] = IntentAnalysis(
                                **analysis_dict,
                                created_at=datetime.now(timezone.utc),
                            )
                        except Exception:
                            # keep resiliency for malformed analysis
                            self.logger.debug(
                                "Failed to parse analyses_buffer during flush",
                                extra={"analyses_buffer": analyses_buffer},
                            )
                        analyses_buffer = ""
                    elif state == GenerationState.RESPONDING and contents_buffer:
                        self.logger.debug(
                            f"Contents buffer: {contents_buffer}\n{'-'*20}\n"
                        )
                        # Use the run id that we recorded when appending
                        # to the contents buffer. The current event's
                        # `run_id` may refer to a different execution (for
                        # example a tool call) so using that would miss the
                        # original streamed content.
                        keyed_run_id = last_content_run_id or run_id
                        content_block = MessageContent(
                            type=MessageContentType.TEXT,
                            text=contents_buffer,
                            created_at=datetime.now(timezone.utc),
                        )
                        message_contents[keyed_run_id] = content_block
                        contents_buffer = ""
                    state = new_state

                res.state = state
                res.prev_state = prev_state

                yield res

        except Exception as e:
            self.logger.error(
                "Workflow execution failed", extra={"error": str(e)}, exc_info=True
            )
            end_time = datetime.now(timezone.utc)
            total_duration = (end_time - start_time).total_seconds() * 1000.0
            yield ChatResponse(
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text="Sorry, I could not complete your request.",
                            created_at=datetime.now(timezone.utc),
                        )
                    ],
                ),
                done=True,
                finish_reason="error",
                total_duration=total_duration,
            )

        if contents_buffer:
            self.logger.debug(
                "Flushing remaining contents_buffer after workflow completion",
                extra={"contents_buffer": contents_buffer},
            )
            # Use the run id that we recorded when appending
            # to the contents buffer. The current event's
            # `run_id` may refer to a different execution (for
            # example a tool call) so using that would miss the
            # original streamed content.
            keyed_run_id = last_content_run_id or run_id
            content_block = MessageContent(
                type=MessageContentType.TEXT,
                text=contents_buffer,
                created_at=datetime.now(timezone.utc),
            )
            message_contents[keyed_run_id] = content_block
            contents_buffer = ""

        if analyses_buffer:
            self.logger.debug(
                "Flushing remaining analyses_buffer after workflow completion",
                extra={"analyses_buffer": analyses_buffer},
            )
            try:
                analysis_dict = json.loads(analyses_buffer)
                analyses[last_analyses_run_id or run_id] = IntentAnalysis(
                    **analysis_dict,
                    created_at=datetime.now(timezone.utc),
                )
            except Exception:
                # keep resiliency for malformed analysis
                self.logger.debug(
                    "Failed to parse analyses_buffer during final flush",
                    extra={"analyses_buffer": analyses_buffer},
                )
            analyses_buffer = ""

        if thoughts_buffer:
            if not raw_think_complete:
                # Model never produced </think> - the buffered content is
                # regular content, not thinking (model doesn't use think tags).
                self.logger.debug(
                    "No </think> found - treating buffered thoughts as regular content",
                    extra={"buffer_len": len(thoughts_buffer)},
                )
                keyed_run_id = last_thoughts_run_id or run_id
                message_contents[keyed_run_id] = MessageContent(
                    type=MessageContentType.TEXT,
                    text=thoughts_buffer,
                    created_at=datetime.now(timezone.utc),
                )
            else:
                self.logger.debug(
                    "Flushing remaining thoughts_buffer after workflow completion",
                    extra={"thoughts_buffer_len": len(thoughts_buffer)},
                )
                thoughts[last_thoughts_run_id or run_id] = Thought(
                    text=thoughts_buffer,
                    created_at=datetime.now(timezone.utc),
                )
            thoughts_buffer = ""

        self.logger.info("Workflow execution completed. Producing final output.")
        final_message = Message(
            role=MessageRole.ASSISTANT,
            content=list(message_contents.values()),
            thoughts=list(thoughts.values()),
            tool_calls=list(tool_calls.values()),
            analyses=list(analyses.values()),
            conversation_id=conversation_id,
            structured_output=structured_content,
        )
        yield ChatResponse(
            message=final_message,
            done=True,
            finish_reason="complete",
            total_duration=(datetime.now(timezone.utc) - start_time).total_seconds()
            * 1000.0,
        )

    def _parse_content_with_reasoning(
        self, content: str | List[str | Dict[str, Any]]
    ) -> tuple[List[str], List[str]]:
        """
        Parse message content into text and reasoning content separately.

        Args:
            content: Content which can be a string or list of strings/dicts

        Returns:
            tuple[List[str], List[str]]: (text_content, reasoning_content)
        """
        text_content = []
        reasoning_content = []

        if isinstance(content, str):
            # Check if it's a JSON string containing reasoning
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and parsed.get("type") == "reasoning":
                    # Extract reasoning content
                    reasoning_text = self._extract_reasoning_text(parsed)
                    if reasoning_text:
                        reasoning_content.append(reasoning_text)
                elif isinstance(parsed, list):
                    # Process as list
                    text_parts, reasoning_parts = self._parse_content_with_reasoning(
                        parsed
                    )
                    text_content.extend(text_parts)
                    reasoning_content.extend(reasoning_parts)
                else:
                    # Regular string content
                    text_content.append(content)
            except (json.JSONDecodeError, TypeError):
                # Not JSON, treat as regular string
                text_content.append(content)
        else:
            # Process list/array content
            for item in content:
                if isinstance(item, str):
                    # Check if string contains reasoning JSON
                    try:
                        parsed = json.loads(item)
                        if (
                            isinstance(parsed, dict)
                            and parsed.get("type") == "reasoning"
                        ):
                            # Extract reasoning content
                            reasoning_text = self._extract_reasoning_text(parsed)
                            if reasoning_text:
                                reasoning_content.append(reasoning_text)
                        else:
                            # Regular string content
                            text_content.append(item)
                    except (json.JSONDecodeError, TypeError):
                        # Not JSON, regular string
                        text_content.append(item)
                elif isinstance(item, dict):
                    # Handle structured content
                    if item.get("type") == "reasoning":
                        # Extract reasoning content
                        reasoning_text = self._extract_reasoning_text(item)
                        if reasoning_text:
                            reasoning_content.append(reasoning_text)
                    elif item.get("type") == "text" and "text" in item:
                        # Extract text field
                        text = item.get("text", "")
                        text_content.append(text)
                    elif "text" in item and not item.get("type"):
                        # Fallback for text without explicit type
                        text = item.get("text", "")
                        text_content.append(text)
                else:
                    # Convert to string if not empty
                    str_item = str(item)
                    # Check if it looks like reasoning JSON
                    if str_item.startswith('{"type": "reasoning"'):
                        try:
                            parsed = json.loads(str_item)
                            reasoning_text = self._extract_reasoning_text(parsed)
                            if reasoning_text:
                                reasoning_content.append(reasoning_text)
                        except json.JSONDecodeError:
                            text_content.append(str_item)
                    else:
                        text_content.append(str_item)

        return text_content, reasoning_content

    def _extract_reasoning_text(self, reasoning_obj: Dict[str, Any]) -> str:
        """
        Extract text from a reasoning object structure.

        Args:
            reasoning_obj: Dictionary containing reasoning data

        Returns:
            str: Extracted reasoning text
        """
        if (
            not isinstance(reasoning_obj, dict)
            or reasoning_obj.get("type") != "reasoning"
        ):
            return ""

        # Extract from summary structure
        summary = reasoning_obj.get("summary", [])
        if isinstance(summary, list):
            text_parts = []
            for summary_item in summary:
                if (
                    isinstance(summary_item, dict)
                    and summary_item.get("type") == "summary_text"
                ):
                    text = summary_item.get("text", "")
                    if text:
                        text_parts.append(text)
                elif isinstance(summary_item, str):
                    text_parts.append(summary_item)
            return "".join(text_parts)
        elif isinstance(summary, str):
            return summary

        # Fallback - look for any text field
        return reasoning_obj.get("text", "")

    def _extract_tool_calls_from_text(
        self, text: str
    ) -> tuple[List[ToolCall], str]:
        """
        Fallback: extract <tool_call> JSON blocks from raw text.

        Some models (especially with Hermes 2 Pro format) sometimes
        embed <tool_call> XML in markdown code fences or regular text
        instead of producing structured tool calls.  This method
        finds them, parses the JSON, and returns ToolCall objects
        plus the remaining text (with tool call blocks removed).

        Returns:
            tuple[List[ToolCall], str]: (extracted tool calls, cleaned text)
        """
        # Match <tool_call>...</tool_call> even inside code fences
        pattern = re.compile(
            r"(?:```\s*)?<tool_call>\s*(\{.*?\})\s*</tool_call>(?:\s*```)?" ,
            re.DOTALL,
        )
        extracted: List[ToolCall] = []
        cleaned = text
        for match in pattern.finditer(text):
            try:
                tc_json = json.loads(match.group(1))
                name = tc_json.get("name", "")
                args = tc_json.get("arguments", tc_json.get("args", {}))
                if isinstance(args, str):
                    args = json.loads(args)
                tc = ToolCall(
                    name=name,
                    args=args,
                    execution_id=f"fallback-{len(extracted)}",
                    created_at=datetime.now(timezone.utc),
                )
                extracted.append(tc)
            except (json.JSONDecodeError, TypeError) as exc:
                self.logger.debug(
                    "Failed to parse fallback tool call",
                    extra={"error": str(exc), "raw": match.group(0)},
                )
        if extracted:
            cleaned = pattern.sub("", text).strip()
        return extracted, cleaned

    def _parse_content(self, content: str | List[str | Dict[str, Any]]) -> List[str]:
        """
        Parse message content into a list of strings.

        Args:
            content: Content which can be a string or list of strings/dicts

        Returns:
            List[str]: Parsed list of string content
        """
        if isinstance(content, str):
            return [content]
        else:
            return [str(c) for c in content]


# Convenience factory functions
def create_executor(
    logger: Optional[Any] = None, context: str = "workflow_executor"
) -> WorkflowExecutor:
    """
    Create a new WorkflowExecutor instance.

    Args:
        logger: Optional logger instance
        context: Default context name

    Returns:
        WorkflowExecutor: New executor instance
    """
    return WorkflowExecutor(logger=logger, default_context=context)


async def stream_workflow(
    initial_state: BaseModel,
    workflow: CompiledStateGraph,
    thread_id: Optional[str] = None,
    config: Optional[RunnableConfig] = None,
    logger: Optional[Any] = None,
    context: str = "workflow_stream",
) -> AsyncIterator[ChatResponse]:
    """
    Convenience function for streaming workflow execution.

    Args:
        workflow: CompiledStateGraph to execute
        initial_state: Initial state for workflow execution
        thread_id: Thread ID for checkpointing
        config: Optional RunnableConfig
        logger: Optional logger instance
        context: Context name for metadata

    Yields:
        Dict[str, Any]: Stream events from workflow execution
    """
    executor = create_executor(logger=logger, context=context)

    async for event in executor.stream_workflow(
        workflow=workflow,
        initial_state=initial_state,
        config=config,
        thread_id=thread_id,
    ):
        yield event
