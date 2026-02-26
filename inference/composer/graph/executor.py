"""
Generic workflow execution module for streaming CompiledStateGraph outputs.

This module provides reusable workflow execution capabilities that can be used
across different graph types and state models, extracting the streaming logic
from ComposerService into a generic, reusable component.
"""

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
from langchain_core.messages import AIMessage, ToolMessage

from composer.constants import STRUCTURED_AGENT_RUNNABLE_NAME
from models import (
    MessageContentType,
    MessageRole,
    Message,
    MessageContent,
    ChatResponse,
    Thought,
    ToolCall,
    GenerationState,
)

from runner.pipelines.llamacpp.chat import ReasoningAwareAIMessageChunk
from utils.logging import llmmllogger


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
        thoughts_buffer = ""
        think_closed = False
        tool_calls: Dict[str, ToolCall] = {}
        thoughts: List[Thought] = []
        message_contents: List[MessageContent] = []
        state: Optional[GenerationState] = None
        prev_state: Optional[GenerationState] = state

        conversation_id = getattr(initial_state, "conversation_id")
        assert conversation_id is not None and isinstance(
            conversation_id, int
        ), "Initial state must have conversation_id"

        def _make_response(**kwargs) -> ChatResponse:
            """Helper to create a ChatResponse with defaults."""
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

        def _strip_think_tags(text: str) -> tuple[str, str]:
            """Split text on </think> boundary. Returns (thinking_part, content_part).
            If no </think> found, returns ('', text) when think already closed,
            or (text, '') when still in think section."""
            nonlocal think_closed
            if "</think>" in text:
                think_closed = True
                before, after = text.split("</think>", 1)
                # Strip <think> prefix if present
                before = before.lstrip()
                if before.startswith("<think>"):
                    before = before[len("<think>") :]
                return before.strip(), after.lstrip("\n")
            if not think_closed:
                # Haven't seen </think> yet; buffer as thinking
                return text, ""
            return "", text

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

                # --- Streaming chunks (token by token) ---
                if event_type in (
                    "on_chat_model_stream",
                    "on_llm_stream",
                ) and isinstance(chunk, AIMessage):
                    # Check for reasoning_content (from ReasoningAwareAIMessageChunk)
                    if hasattr(chunk, "reasoning_content"):
                        rc = cast(ReasoningAwareAIMessageChunk, chunk).reasoning_content
                        if rc and rc.strip():
                            thoughts_buffer += rc
                            new_state = GenerationState.THINKING
                            res = _make_response(
                                message_kwargs={"thoughts": [Thought(text=rc)]}
                            )
                            if new_state != state:
                                self.logger.debug(
                                    f"State transition: {state} -> {new_state}"
                                )
                                state = new_state
                            prev_state = state
                            yield res
                            continue

                    # Regular content - handle <think> tags
                    if chunk.content:
                        text_parts = self._parse_content(chunk.content)
                        for text in text_parts:
                            thinking_part, content_part = _strip_think_tags(text)

                            if thinking_part:
                                thoughts_buffer += thinking_part
                                new_state = GenerationState.THINKING
                                res = _make_response(
                                    message_kwargs={
                                        "thoughts": [Thought(text=thinking_part)]
                                    }
                                )
                                if new_state != state:
                                    self.logger.debug(
                                        f"State transition: {state} -> {new_state}"
                                    )
                                    state = new_state
                                prev_state = state
                                yield res

                            if content_part:
                                contents_buffer += content_part
                                new_state = GenerationState.RESPONDING
                                res = _make_response(
                                    message_kwargs={
                                        "content": [
                                            MessageContent(
                                                type=MessageContentType.TEXT,
                                                text=content_part,
                                            )
                                        ]
                                    }
                                )
                                if new_state != state:
                                    self.logger.debug(
                                        f"State transition: {state} -> {new_state}"
                                    )
                                    state = new_state
                                prev_state = state
                                yield res

                # --- Model generation complete ---
                elif event_type in ("on_chat_model_end", "on_llm_end"):
                    if isinstance(output, AIMessage):
                        # Extract proxy tool calls from bind_tools()
                        if hasattr(output, "tool_calls") and output.tool_calls:
                            res = _make_response()
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

                        # Extract content from non-streaming end events
                        # (when streaming is disabled for tool calling)
                        if output.content and not contents_buffer:
                            text_parts = self._parse_content(output.content)
                            full_text = "".join(text_parts)
                            thinking_part, content_part = _strip_think_tags(full_text)

                            if thinking_part:
                                thoughts_buffer += thinking_part

                            if content_part:
                                contents_buffer += content_part
                                new_state = GenerationState.RESPONDING
                                res = _make_response(
                                    message_kwargs={
                                        "content": [
                                            MessageContent(
                                                type=MessageContentType.TEXT,
                                                text=content_part,
                                            )
                                        ]
                                    }
                                )
                                if new_state != state:
                                    self.logger.debug(
                                        f"State transition: {state} -> {new_state}"
                                    )
                                    state = new_state
                                prev_state = state
                                yield res

                        md = output.response_metadata or {}
                        reason = md.get("finish_reason") or "unknown"
                        has_tc = bool(
                            hasattr(output, "tool_calls") and output.tool_calls
                        )
                        self.logger.debug(
                            "Model generation completed",
                            extra={
                                "finish_reason": reason,
                                "has_tool_calls": has_tc,
                                "content_len": len(contents_buffer),
                            },
                        )

                # --- Structured output (grammar mode) ---
                elif (
                    event_type == "on_chain_end"
                    and event_name == STRUCTURED_AGENT_RUNNABLE_NAME
                ):
                    new_state = GenerationState.FORMATTING
                    if isinstance(output, BaseModel):
                        output = output.model_dump()
                    res = _make_response()
                    assert res.message
                    res.message.structured_output = output

                # --- Server-side tool execution ---
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
                    res = _make_response(message_kwargs={"tool_calls": [tc]})
                    yield res

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

        # Build final accumulated message

        # If think never closed, the buffered "thoughts" are actually content
        if thoughts_buffer and not think_closed:
            contents_buffer = thoughts_buffer + contents_buffer
            thoughts_buffer = ""

        # If think closed but model produced NO content and NO tool calls,
        # the thoughts are the only output — promote them to content so
        # the response isn't empty.
        if thoughts_buffer and not contents_buffer and not tool_calls:
            self.logger.debug(
                "No content or tool calls after thinking — promoting thoughts to content",
                extra={"thoughts_len": len(thoughts_buffer)},
            )
            contents_buffer = thoughts_buffer
            thoughts_buffer = ""

        if contents_buffer:
            message_contents.append(
                MessageContent(
                    type=MessageContentType.TEXT,
                    text=contents_buffer,
                    created_at=datetime.now(timezone.utc),
                )
            )

        if thoughts_buffer:
            thoughts.append(
                Thought(text=thoughts_buffer, created_at=datetime.now(timezone.utc))
            )

        self.logger.info("Workflow execution completed. Producing final output.")
        final_message = Message(
            role=MessageRole.ASSISTANT,
            content=message_contents,
            thoughts=thoughts,
            tool_calls=list(tool_calls.values()),
            conversation_id=conversation_id,
        )
        yield ChatResponse(
            message=final_message,
            done=True,
            finish_reason="complete",
            total_duration=(datetime.now(timezone.utc) - start_time).total_seconds()
            * 1000.0,
        )

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
