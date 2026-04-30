"""
Generic workflow execution module for streaming CompiledStateGraph outputs.
This module provides reusable workflow execution capabilities that can be used
across different graph types and state models, extracting the streaming logic
from ComposerService into a generic, reusable component.
"""

import asyncio
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Union,
)
from datetime import datetime, timezone
from pydantic import BaseModel
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import AIMessage, ToolMessage
from constants import STRUCTURED_AGENT_RUNNABLE_NAME, TOOL_NODE_NAME
from graph.state import ServerToolEvent
from models import (
    MessageContentType,
    MessageRole,
    Message,
    MessageContent,
    ChatResponse,
    ToolCall,
    GenerationState,
)
from utils.logging import llmmllogger, serialize_event_data

from .content_parser import (
    parse_content,
    RawToolCallStreamBuffer,
)
from .tool_call_parser import RawToolCallParser, _RAW_TOOL_CALL_RE


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
        self.content_parser = RawToolCallParser()

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
        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 25,
        }
        if additional_config:
            config.setdefault("configurable", {}).update(additional_config)
        return config

    def _make_response(
        self,
        conversation_id: int,
        state: Optional[GenerationState],
        prev_state: Optional[GenerationState],
        **kwargs,
    ) -> ChatResponse:
        """
        Create a ChatResponse with defaults.

        Args:
            conversation_id: Conversation identifier
            state: Current generation state
            prev_state: Previous generation state
            **kwargs: Additional response kwargs

        Returns:
            ChatResponse: Constructed response object
        """
        msg_kwargs = {
            "role": MessageRole.ASSISTANT,
            "content": [],
            "thoughts": [],
            "tool_calls": [],
            "conversation_id": conversation_id,
        }
        msg_kwargs.update(kwargs.pop("message_kwargs", {}))
        return ChatResponse(
            done=False,
            message=Message(**msg_kwargs),
            state=state,
            prev_state=prev_state,
            **kwargs,
        )

    def _emit_tool_calls_from_raw(
        self,
        raw_blocks: List[str],
        tool_calls: Dict[str, ToolCall],
        conversation_id: int,
        state: Optional[GenerationState],
        prev_state: Optional[GenerationState],
    ):
        """
        Parse a list of complete raw tool-call XML blocks, register them in
        tool_calls, and return (response, new_state) if any were parsed, or
        (None, state) if the list was empty / unparseable.
        """
        parsed: List[ToolCall] = []
        for block in raw_blocks:
            _, tcs = self.content_parser.strip_raw_tool_calls(block)
            parsed.extend(tcs)

        if not parsed:
            return None, state

        tc_res = self._make_response(conversation_id, state, prev_state)
        assert tc_res.message and tc_res.message.tool_calls is not None
        for tc in parsed:
            tc_key = tc.execution_id or tc.name
            tool_calls[tc_key] = tc
            tc_res.message.tool_calls.append(tc)

        new_state = GenerationState.EXECUTING
        return tc_res, new_state

    async def stream_workflow(
        self,
        workflow: CompiledStateGraph,
        initial_state: BaseModel,
        config: Optional[RunnableConfig] = None,
        thread_id: Optional[str] = None,
    ) -> AsyncIterator[Union[ChatResponse, ServerToolEvent]]:
        """
        Execute a compiled workflow with streaming output.

        Streams ChatResponse events for each meaningful model chunk,
        then yields a final done=True event with accumulated results.

        Args:
            workflow: CompiledStateGraph to execute
            initial_state: Initial state for workflow execution
            config: Optional RunnableConfig
            thread_id: Thread ID for checkpointing (used if config is None)

        Yields:
            ChatResponse: Stream events from workflow execution
        """
        start_time = datetime.now(timezone.utc)
        contents_buffer = ""
        tool_calls: Dict[str, ToolCall] = {}
        message_contents: List[MessageContent] = []
        state: Optional[GenerationState] = None
        prev_state: Optional[GenerationState] = state
        # Track the model's actual finish reason (stop, tool_calls, length, etc.)
        # so the completion service can distinguish natural stops from truncations.
        model_finish_reason: str = "complete"
        # Track how many server_tool_events we've already yielded to avoid
        # duplicates (the state field accumulates via operator.add).
        server_tool_events_yielded = 0
        conversation_id = getattr(initial_state, "conversation_id")
        assert conversation_id is not None and isinstance(
            conversation_id, int
        ), "Initial state must have conversation_id"

        # Per-stream buffer that holds back raw tool-call XML until the
        # closing tag arrives, preventing partial XML from leaking to the
        # client as garbled text content.
        tc_stream_buf = RawToolCallStreamBuffer()
        # Flag: true when the stream has seen structured tool_call_chunks
        # from LangChain.  Used to suppress duplicate content-as-text.
        streaming_has_tool_call_chunks = False

        try:
            # Prepare state dict
            if isinstance(initial_state, dict):
                state_dict = initial_state
            elif hasattr(initial_state, "model_dump"):
                state_dict = initial_state.model_dump()
            else:
                raise ValueError(
                    f"State type {type(initial_state)} must be dict or have model_dump method"
                )
            if config is None and thread_id is not None:
                config = self.create_thread_config(thread_id)

            async for event in workflow.astream_events(
                state_dict,
                config=config,
                version="v2",
            ):
                data = event.get("data", {})
                event_type = event.get("event", "")
                chunk = data.get("chunk")
                output = data.get("output")
                event_name = event.get("name", "")
                run_id = event.get("run_id", "")
                new_state = state

                # ----------------------------------------------------------------
                # Streaming chunks (token by token)
                # ----------------------------------------------------------------
                if event_type in (
                    "on_chat_model_stream",
                    "on_llm_stream",
                ) and isinstance(chunk, AIMessage):

                    # -- Streaming tool_call_chunks --
                    # When LangChain parses structured tool calls from the
                    # stream, it puts them in tool_call_chunks (accumulated
                    # into output.tool_calls at on_chat_model_end).  If the
                    # model also emits the tool call as content text, the
                    # stream buffer will catch it.  Flag that we've seen
                    # structured chunks so we can suppress the duplicate text.
                    if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                        streaming_has_tool_call_chunks = True

                    # -- Regular content tokens --
                    if chunk.content:
                        text_parts = parse_content(chunk.content)
                        for raw_text in text_parts:
                            # Pass content through raw - no think tag stripping
                            safe_text, complete_blocks = tc_stream_buf.feed(raw_text)

                            # Emit safe text (content that precedes any tool-call
                            # XML, or plain content with no tool-call markers).
                            if safe_text:
                                contents_buffer += safe_text
                                new_state = GenerationState.RESPONDING
                                res = self._make_response(
                                    conversation_id,
                                    state,
                                    prev_state,
                                    message_kwargs={
                                        "content": [
                                            MessageContent(
                                                type=MessageContentType.TEXT,
                                                text=safe_text,
                                            )
                                        ]
                                    },
                                )
                                if new_state != state:
                                    self.logger.debug(
                                        f"State transition: {state} -> {new_state}"
                                    )
                                    state = new_state
                                prev_state = state
                                yield res

                            # Emit any complete tool-call blocks that were just
                            # closed by this chunk.
                            if complete_blocks:
                                tc_res, new_state = self._emit_tool_calls_from_raw(
                                    complete_blocks,
                                    tool_calls,
                                    conversation_id,
                                    state,
                                    prev_state,
                                )
                                if tc_res is not None:
                                    if new_state != state:
                                        self.logger.debug(
                                            f"State transition: {state} -> {new_state}"
                                        )
                                        state = new_state
                                    prev_state = state
                                    yield tc_res

                # ----------------------------------------------------------------
                # Model generation complete
                # ----------------------------------------------------------------
                elif event_type in ("on_chat_model_end", "on_llm_end"):
                    if isinstance(output, AIMessage):
                        # Flush the stream buffer — if the stream ended while we
                        # were still inside a tool-call block (e.g. truncation),
                        # treat the incomplete XML as a complete block so we can
                        # attempt to parse it rather than silently dropping it.
                        flush_text, flush_blocks = tc_stream_buf.flush()
                        if flush_text:
                            contents_buffer += flush_text
                        if flush_blocks:
                            self.logger.debug(
                                "Flushing incomplete tool-call block at stream end",
                                extra={"block_count": len(flush_blocks)},
                            )
                            tc_res, new_state = self._emit_tool_calls_from_raw(
                                flush_blocks,
                                tool_calls,
                                conversation_id,
                                state,
                                prev_state,
                            )
                            if tc_res is not None:
                                if new_state != state:
                                    self.logger.debug(
                                        f"State transition: {state} -> {new_state}"
                                    )
                                    state = new_state
                                prev_state = state
                                yield tc_res

                        # Extract structured tool calls from bind_tools() output.
                        if hasattr(output, "tool_calls") and output.tool_calls:
                            res = self._make_response(
                                conversation_id, state, prev_state
                            )
                            assert res.message and res.message.tool_calls is not None
                            for tc_data in output.tool_calls:
                                tc_id = tc_data.get("id") or run_id
                                if tc_id not in tool_calls:
                                    tc = ToolCall(
                                        name=tc_data.get("name", ""),
                                        args=tc_data.get("args", {}),
                                        execution_id=tc_id,
                                        created_at=datetime.now(timezone.utc),
                                    )
                                    tool_calls[tc_id] = tc
                                    res.message.tool_calls.append(tc)
                            new_state = GenerationState.EXECUTING
                            if new_state != state:
                                self.logger.debug(
                                    f"State transition: {state} -> {new_state}"
                                )
                                state = new_state
                            prev_state = state
                            yield res

                        # For non-streaming completions (e.g. grammar / tool mode
                        # where the whole response arrives in on_llm_end), extract
                        # content now.  Skip when tool calls are present — the
                        # content field may contain raw tool-call markup that must
                        # not leak as visible text.  Also skip when we already
                        # saw structured tool_call_chunks during streaming.
                        has_end_tc = (
                            bool(hasattr(output, "tool_calls") and output.tool_calls)
                            or streaming_has_tool_call_chunks
                        )
                        self.logger.debug(
                            "on_llm_end event",
                            extra={
                                "has_output_content": bool(output.content),
                                "contents_buffer_len": len(contents_buffer),
                                "has_end_tc": has_end_tc,
                                "will_extract": bool(
                                    output.content
                                    and not contents_buffer
                                    and not has_end_tc
                                ),
                            },
                        )
                        if output.content and not contents_buffer and not has_end_tc:
                            text_parts = parse_content(output.content)
                            full_text = "".join(text_parts).strip()

                            # Strip orphaned think tags that the streaming path
                            # may have already consumed.  When --reasoning-budget
                            # 0 forces the model to skip thinking, a bare
                            # </think> can appear as the entire output; the
                            # streaming handler silently absorbs it but
                            # contents_buffer stays empty.  Without this cleanup
                            # the tag leaks as literal content, poisoning the
                            # conversation history and causing the model to emit
                            # EOS on subsequent invocations.
                            if "</think>" in full_text:
                                full_text = full_text.split("</think>", 1)[-1].strip()
                            if full_text.strip() == "<think>":
                                full_text = ""

                            # Always try to parse — the enhanced parser
                            # handles XML tags, bare JSON, code blocks, and
                            # Mistral-style [TOOL_CALLS] format.
                            content_part, raw_tcs = (
                                self.content_parser.strip_raw_tool_calls(full_text)
                            )

                            if content_part:
                                self.logger.info(
                                    "Yielding content from on_llm_end",
                                    extra={
                                        "content_len": len(content_part),
                                        "content_preview": content_part[:200],
                                    },
                                )
                                contents_buffer += content_part
                                new_state = GenerationState.RESPONDING
                                res = self._make_response(
                                    conversation_id,
                                    state,
                                    prev_state,
                                    message_kwargs={
                                        "content": [
                                            MessageContent(
                                                type=MessageContentType.TEXT,
                                                text=content_part,
                                            )
                                        ]
                                    },
                                )
                                if new_state != state:
                                    self.logger.debug(
                                        f"State transition: {state} -> {new_state}"
                                    )
                                    state = new_state
                                prev_state = state
                                yield res

                            if raw_tcs:
                                tc_res, new_state = self._emit_tool_calls_from_raw(
                                    # raw_tcs are already parsed ToolCall objects
                                    # from strip_raw_tool_calls; re-wrap as "blocks"
                                    # isn't right here — emit them directly.
                                    [],  # blocks already parsed below
                                    tool_calls,
                                    conversation_id,
                                    state,
                                    prev_state,
                                )
                                # strip_raw_tool_calls returns ToolCall objects
                                # directly, so register and emit them without
                                # going through _emit_tool_calls_from_raw.
                                tc_res = self._make_response(
                                    conversation_id, state, prev_state
                                )
                                assert (
                                    tc_res.message
                                    and tc_res.message.tool_calls is not None
                                )
                                for tc in raw_tcs:
                                    tc_key = tc.execution_id or tc.name
                                    tool_calls[tc_key] = tc
                                    tc_res.message.tool_calls.append(tc)
                                new_state = GenerationState.EXECUTING
                                if new_state != state:
                                    self.logger.debug(
                                        f"State transition: {state} -> {new_state}"
                                    )
                                    state = new_state
                                prev_state = state
                                yield tc_res

                        md = output.response_metadata or {}
                        reason = md.get("finish_reason") or "unknown"
                        # Normalize: llama.cpp returns "tool_calls" (plural)
                        # but ChatResponse expects "tool_call" (singular).
                        if reason == "tool_calls":
                            reason = "tool_call"
                        model_finish_reason = reason
                        self.logger.debug(
                            "Model generation completed",
                            extra={
                                "finish_reason": reason,
                                "has_tool_calls": has_end_tc,
                                "content_len": len(contents_buffer),
                            },
                        )

                # ----------------------------------------------------------------
                # Structured output (grammar mode)
                # ----------------------------------------------------------------
                elif (
                    event_type == "on_chain_end"
                    and event_name == STRUCTURED_AGENT_RUNNABLE_NAME
                ):
                    new_state = GenerationState.FORMATTING
                    if isinstance(output, BaseModel):
                        output = output.model_dump()
                    res = self._make_response(conversation_id, state, prev_state)
                    assert res.message
                    res.message.structured_output = output

                # ----------------------------------------------------------------
                # ServerToolNode completed — yield ServerToolEvents so the
                # router can emit SSE content blocks at iteration boundaries.
                # ----------------------------------------------------------------
                elif event_type == "on_chain_end" and event_name == TOOL_NODE_NAME:
                    # output is the state dict returned by ServerToolNode
                    raw_events: list = []
                    if isinstance(output, dict):
                        raw_events = output.get("server_tool_events", [])
                    elif output is not None and hasattr(output, "server_tool_events"):
                        raw_events = output.server_tool_events  # type: ignore[union-attr]

                    # Only yield events we haven't seen yet (the state
                    # accumulates via operator.add across iterations).
                    new_events = raw_events[server_tool_events_yielded:]
                    for evt in new_events:
                        tc = (
                            evt.get("tool_call")
                            if isinstance(evt, dict)
                            else getattr(evt, "tool_call", None)
                        )
                        result: str = (
                            evt.get("result_text", "")
                            if isinstance(evt, dict)
                            else getattr(evt, "result_text", "")
                        ) or ""
                        canonical: str = (
                            evt.get("canonical_name", "")
                            if isinstance(evt, dict)
                            else getattr(evt, "canonical_name", "")
                        ) or ""
                        if tc is not None:
                            yield ServerToolEvent(
                                tool_call=tc,
                                result_text=result,
                                canonical_name=canonical,
                            )
                            server_tool_events_yielded += 1

                # ----------------------------------------------------------------
                # Server-side tool execution (LangGraph built-in ToolNode)
                # ----------------------------------------------------------------
                elif event_type.endswith("_tool_start"):
                    new_state = GenerationState.EXECUTING
                    tool_calls[run_id] = ToolCall(
                        name=event_name,
                        args=data.get("input", {}),
                        execution_id=run_id,
                        created_at=datetime.now(timezone.utc),
                    )

                elif event_type.endswith("_tool_end") and isinstance(
                    output, ToolMessage
                ):
                    tc = tool_calls.get(run_id)
                    if tc is None:
                        tc = ToolCall(
                            name=event_name,
                            args=data.get("input", {}),
                            execution_id=run_id,
                        )
                    tc.success = True
                    tc.result_data = output.model_dump()
                    tc.created_at = datetime.now(timezone.utc)
                    tool_calls[run_id] = tc
                    res = self._make_response(
                        conversation_id,
                        state,
                        prev_state,
                        message_kwargs={"tool_calls": [tc]},
                    )
                    yield res

        except asyncio.CancelledError:
            self.logger.warning("Workflow cancelled (client disconnect)")
            raise

        except Exception as e:
            self.logger.error(
                "Workflow execution failed", extra={"error": str(e)}, exc_info=True
            )
            total_duration = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000.0
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
            return  # Don't fall through to final-message assembly (empty response would overwrite error)

        # --------------------------------------------------------------------
        # Build final accumulated message - pass through raw
        # --------------------------------------------------------------------
        if contents_buffer:
            message_contents.append(
                MessageContent(
                    type=MessageContentType.TEXT,
                    text=contents_buffer,
                    created_at=datetime.now(timezone.utc),
                )
            )

        self.logger.info("Workflow execution completed. Producing final output.")
        self.logger.info(
            "Final message construction",
            extra={
                "contents_buffer_len": len(contents_buffer),
                "contents_buffer_preview": (
                    contents_buffer[:200] if contents_buffer else ""
                ),
                "message_contents_len": len(message_contents),
                "total_tool_calls": len(tool_calls),
            },
        )

        if not contents_buffer and not tool_calls:
            self.logger.warning(
                "Model produced empty response — no content or tool calls.",
            )

        final_message = Message(
            role=MessageRole.ASSISTANT,
            content=message_contents,
            thoughts=[],
            tool_calls=list(tool_calls.values()),
            conversation_id=conversation_id,
        )
        yield ChatResponse(
            message=final_message,
            done=True,
            finish_reason=model_finish_reason,
            total_duration=(datetime.now(timezone.utc) - start_time).total_seconds()
            * 1000.0,
        )


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------


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
) -> AsyncIterator[Union[ChatResponse, ServerToolEvent]]:
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
        ChatResponse or ServerToolEvent: Stream events from workflow execution
    """
    executor = create_executor(logger=logger, context=context)
    async for event in executor.stream_workflow(
        workflow=workflow,
        initial_state=initial_state,
        config=config,
        thread_id=thread_id,
    ):
        yield event
