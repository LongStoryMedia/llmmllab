"""
CompletionService — shared workflow execution, retry, and continuation logic.

This is the core service that all three LLM endpoints delegate to.  It owns:
  • Building and executing a composer workflow
  • Single-retry on empty responses
  • Nudge prompt when retries also fail
  • Tool-continuation check (model produced text but no tool calls)
  • Filtering server-tool calls from the final response

The service yields ``ChatResponse | ServerToolEvent`` objects — the routers
are responsible only for formatting those into the wire protocol (SSE chunks
in Anthropic/OpenAI format, raw JSON for llmmllab).
"""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Optional, Union

from composer_init import (
    compose_workflow,
    create_initial_state,
    execute_workflow,
    get_graph_builder,
)
from graph.state import ServerToolEvent
from graph.workflows.factory import WorkFlowType
from models.chat_response import ChatResponse
from models.message import Message, MessageContent, MessageContentType, MessageRole
from models.tool_call import ToolCall
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="completion_service")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from config import ENABLE_TOOL_CONTINUATION

_CONTINUATION_ENABLED = ENABLE_TOOL_CONTINUATION

_CONTINUATION_PROMPT = (
    "You described using a tool but did not actually call one. "
    "Call the appropriate tool now. Do not describe what you will do — invoke the tool directly."
)

_EMPTY_RESPONSE_NUDGE = (
    "Your response didn't produce any output. Did you mean to say something "
    "or use a tool? If so, continue. Otherwise, simply respond with 'done' "
    "and nothing else."
)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class CompletionResult:
    """Accumulated result from a non-streaming completion."""

    chat_response: Optional[ChatResponse] = None
    output_tokens: int = 0

    @property
    def has_content(self) -> bool:
        return bool(
            self.chat_response
            and self.chat_response.message
            and self.chat_response.message.content
            and any(
                c.text
                for c in self.chat_response.message.content
                if c.type == MessageContentType.TEXT and c.text
            )
        )

    @property
    def has_tool_calls(self) -> bool:
        return bool(
            self.chat_response
            and self.chat_response.message
            and self.chat_response.message.tool_calls
        )

    @property
    def is_error(self) -> bool:
        return bool(self.chat_response and self.chat_response.finish_reason == "error")


@dataclass
class StreamAccumulator:
    """Mutable state accumulated while streaming events to the router."""

    has_content: bool = False
    has_tool_calls: bool = False
    is_error: bool = False
    finish_reason: str = ""
    final_tool_calls: list[ToolCall] = field(default_factory=list)
    final_content: str = ""
    output_tokens: int = 0
    input_tokens: int = 0


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class CompletionService:
    """Shared workflow execution and resilience logic.

    All methods are static / stateless — the service exists as an
    organisational namespace, not a stateful singleton.
    """

    # ------------------------------------------------------------------
    # Core workflow helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _build_and_run(
        user_id: str,
        messages: list[Message],
        model_name: str,
        workflow_type: WorkFlowType,
        conversation_id: int = 0,
        client_tools: list | None = None,
        tool_choice: str | None = None,
        server_tool_names: set[str] | None = None,
    ) -> AsyncIterator[Union[ChatResponse, ServerToolEvent]]:
        """Build a composer workflow and yield its events."""
        builder = await get_graph_builder(workflow_type, user_id)
        workflow = await compose_workflow(
            user_id=user_id,
            builder=builder,
            model_name=model_name,
            client_tools=client_tools,
            tool_choice=tool_choice,
            server_tool_names=server_tool_names or None,
        )
        initial_state = await create_initial_state(
            user_id, conversation_id, builder, messages
        )
        async for event in execute_workflow(initial_state, workflow):
            yield event

    # ------------------------------------------------------------------
    # Streaming path  (yields raw events; router formats SSE)
    # ------------------------------------------------------------------

    @staticmethod
    async def stream_completion(
        user_id: str,
        messages: list[Message],
        model_name: str,
        workflow_type: WorkFlowType = WorkFlowType.IDE,
        conversation_id: int = 0,
        client_tools: list | None = None,
        tool_choice: str | None = None,
        server_tool_names: set[str] | None = None,
    ) -> AsyncIterator[tuple[Union[ChatResponse, ServerToolEvent], StreamAccumulator]]:
        """Execute a workflow and yield ``(event, accumulator)`` pairs.

        The *accumulator* is mutated in-place as events arrive so that the
        router can inspect it at any time (e.g. to decide whether to start
        a text block).

        After the primary pass completes, the method transparently performs
        continuation and retry logic and continues yielding events from
        those secondary passes.
        """
        acc = StreamAccumulator()

        try:
            # ---------- primary pass ----------
            async for event in CompletionService._build_and_run(
                user_id,
                messages,
                model_name,
                workflow_type,
                conversation_id,
                client_tools,
                tool_choice,
                server_tool_names,
            ):
                if isinstance(event, ServerToolEvent):
                    yield event, acc
                    continue

                if event.done:
                    if event.finish_reason == "error":
                        acc.is_error = True
                    if event.finish_reason:
                        acc.finish_reason = event.finish_reason
                    if event.message and event.message.tool_calls:
                        acc.final_tool_calls = event.message.tool_calls
                        acc.has_tool_calls = True
                    if event.message and event.message.content:
                        parts = [
                            c.text
                            for c in event.message.content
                            if c.type == MessageContentType.TEXT and c.text
                        ]
                        acc.final_content = "".join(parts)
                    if event.prompt_eval_count:
                        acc.input_tokens = int(event.prompt_eval_count)
                    if event.eval_count:
                        acc.output_tokens = int(event.eval_count)
                    yield event, acc
                    continue

                # Live delta — let the router stream it
                if event.message and event.message.content:
                    for part in event.message.content:
                        if part.type == MessageContentType.TEXT and part.text:
                            acc.has_content = True
                yield event, acc

            # Filter server tool calls out of final_tool_calls
            if server_tool_names and acc.final_tool_calls:
                acc.final_tool_calls = [
                    tc
                    for tc in acc.final_tool_calls
                    if tc.name not in server_tool_names
                ]
                acc.has_tool_calls = bool(acc.final_tool_calls)

            # ---------- continuation check ----------
            # Skip when the model naturally stopped (finish_reason="stop") —
            # it intentionally chose not to call a tool and forcing one
            # creates an infinite loop where the model keeps saying "done"
            # but gets coerced into unnecessary tool calls.
            # Also skip on "length" — the model was cut off mid-response
            # by the token limit, not intentionally avoiding tool calls.
            if (
                _CONTINUATION_ENABLED
                and not acc.has_tool_calls
                and client_tools
                and (acc.has_content or acc.final_content)
                and acc.finish_reason not in ("stop", "length")
            ):
                accumulated_text = acc.final_content or ""
                logger.info(
                    "Model produced text without tool calls — sending single continuation check",
                    extra={
                        "content_len": len(accumulated_text),
                        "content_preview": accumulated_text[:200],
                    },
                )
                continuation_messages = list(messages) + [
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT, text=accumulated_text
                            )
                        ],
                    ),
                    Message(
                        role=MessageRole.USER,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT, text=_CONTINUATION_PROMPT
                            )
                        ],
                    ),
                ]
                async for event in CompletionService._build_and_run(
                    user_id,
                    continuation_messages,
                    model_name,
                    workflow_type,
                    conversation_id,
                    client_tools,
                    "auto",
                    server_tool_names,
                ):
                    if isinstance(event, ServerToolEvent):
                        continue
                    if event.done:
                        if event.message and event.message.tool_calls:
                            acc.final_tool_calls = event.message.tool_calls
                            acc.has_tool_calls = True
                        if event.eval_count:
                            acc.output_tokens += int(event.eval_count)
                        continue

            # ---------- empty-response retry ----------
            if (
                not acc.has_content
                and not acc.has_tool_calls
                and not acc.final_content
                and not acc.is_error
            ):
                logger.warning(
                    "Model produced empty response — retrying with same messages",
                    extra={"model": model_name},
                )
                async for event in CompletionService._build_and_run(
                    user_id,
                    messages,
                    model_name,
                    workflow_type,
                    conversation_id,
                    client_tools,
                    tool_choice,
                    server_tool_names,
                ):
                    if isinstance(event, ServerToolEvent):
                        continue
                    if event.done:
                        if event.message and event.message.tool_calls:
                            acc.final_tool_calls = event.message.tool_calls
                            acc.has_tool_calls = True
                        if event.message and event.message.content:
                            parts = [
                                c.text
                                for c in event.message.content
                                if c.type == MessageContentType.TEXT and c.text
                            ]
                            acc.final_content = "".join(parts)
                        if event.eval_count:
                            acc.output_tokens += int(event.eval_count)
                        continue

                    # Live delta from retry
                    if event.message and event.message.content:
                        for part in event.message.content:
                            if part.type == MessageContentType.TEXT and part.text:
                                acc.has_content = True
                    yield event, acc

                # ---------- nudge if retry also empty ----------
                if (
                    not acc.has_content
                    and not acc.has_tool_calls
                    and not acc.final_content
                ):
                    logger.warning(
                        "Retry also produced empty response — sending nudge prompt",
                        extra={"model": model_name},
                    )
                    nudge_messages = list(messages) + [
                        Message(
                            role=MessageRole.USER,
                            content=[
                                MessageContent(
                                    type=MessageContentType.TEXT,
                                    text=_EMPTY_RESPONSE_NUDGE,
                                )
                            ],
                        ),
                    ]
                    async for event in CompletionService._build_and_run(
                        user_id,
                        nudge_messages,
                        model_name,
                        workflow_type,
                        conversation_id,
                        client_tools,
                        "auto",
                        server_tool_names,
                    ):
                        if isinstance(event, ServerToolEvent):
                            continue
                        if event.done:
                            if event.message and event.message.tool_calls:
                                acc.final_tool_calls = event.message.tool_calls
                                acc.has_tool_calls = True
                            if event.message and event.message.content:
                                parts = [
                                    c.text
                                    for c in event.message.content
                                    if c.type == MessageContentType.TEXT and c.text
                                ]
                                acc.final_content = "".join(parts)
                            if event.eval_count:
                                acc.output_tokens += int(event.eval_count)
                            continue

                        if event.message and event.message.content:
                            for part in event.message.content:
                                if part.type == MessageContentType.TEXT and part.text:
                                    acc.has_content = True
                        yield event, acc

        except asyncio.CancelledError:
            logger.warning("Stream cancelled (client disconnect) — stopping retries")
            return

    # ------------------------------------------------------------------
    # Non-streaming path
    # ------------------------------------------------------------------

    @staticmethod
    async def run_completion(
        user_id: str,
        messages: list[Message],
        model_name: str,
        workflow_type: WorkFlowType = WorkFlowType.IDE,
        conversation_id: int = 0,
        client_tools: list | None = None,
        tool_choice: str | None = None,
        server_tool_names: set[str] | None = None,
    ) -> CompletionResult:
        """Execute a workflow and return the final accumulated result.

        Handles continuation, empty-response retry, and nudge logic
        identically to the streaming path.
        """
        result = CompletionResult()

        # ---------- primary pass ----------
        async for event in CompletionService._build_and_run(
            user_id,
            messages,
            model_name,
            workflow_type,
            conversation_id,
            client_tools,
            tool_choice,
            server_tool_names,
        ):
            if isinstance(event, ServerToolEvent):
                continue
            if event.done and event.message:
                result.chat_response = event

        if result.chat_response is None:
            return result

        # Filter server tool calls
        if (
            server_tool_names
            and result.chat_response.message
            and result.chat_response.message.tool_calls
        ):
            result.chat_response.message.tool_calls = [
                tc
                for tc in result.chat_response.message.tool_calls
                if tc.name not in server_tool_names
            ]

        # ---------- continuation check ----------
        # Skip when the model naturally stopped or was cut off by token
        # limit — see streaming path comment.
        skip_continuation = (
            result.chat_response
            and result.chat_response.finish_reason in ("stop", "length")
        )
        if (
            _CONTINUATION_ENABLED
            and not result.has_tool_calls
            and client_tools
            and result.has_content
            and not skip_continuation
        ):
            accumulated_text = "".join(
                c.text
                for c in result.chat_response.message.content  # type: ignore
                if c.type == MessageContentType.TEXT and c.text
            )
            if accumulated_text:
                logger.info(
                    "Non-streaming: model produced text without tool calls — sending single continuation check",
                    extra={
                        "content_len": len(accumulated_text),
                        "content_preview": accumulated_text[:200],
                    },
                )
                continuation_messages = list(messages) + [
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT, text=accumulated_text
                            )
                        ],
                    ),
                    Message(
                        role=MessageRole.USER,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT, text=_CONTINUATION_PROMPT
                            )
                        ],
                    ),
                ]
                async for event in CompletionService._build_and_run(
                    user_id,
                    continuation_messages,
                    model_name,
                    workflow_type,
                    conversation_id,
                    client_tools,
                    "auto",
                    server_tool_names,
                ):
                    if isinstance(event, ServerToolEvent):
                        continue
                    if event.done and event.message:
                        if event.message.tool_calls:
                            result.chat_response = event

        # ---------- empty-response retry ----------
        if not result.has_content and not result.has_tool_calls and not result.is_error:
            logger.warning(
                "Non-streaming: model produced empty response — retrying",
                extra={"model": model_name},
            )
            async for event in CompletionService._build_and_run(
                user_id,
                messages,
                model_name,
                workflow_type,
                conversation_id,
                client_tools,
                tool_choice,
                server_tool_names,
            ):
                if isinstance(event, ServerToolEvent):
                    continue
                if event.done and event.message:
                    result.chat_response = event

            # ---------- nudge ----------
            if not result.has_content and not result.has_tool_calls:
                logger.warning(
                    "Non-streaming: retry also empty — sending nudge prompt",
                    extra={"model": model_name},
                )
                nudge_messages = list(messages) + [
                    Message(
                        role=MessageRole.USER,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT, text=_EMPTY_RESPONSE_NUDGE
                            )
                        ],
                    ),
                ]
                async for event in CompletionService._build_and_run(
                    user_id,
                    nudge_messages,
                    model_name,
                    workflow_type,
                    conversation_id,
                    client_tools,
                    "auto",
                    server_tool_names,
                ):
                    if isinstance(event, ServerToolEvent):
                        continue
                    if event.done and event.message:
                        result.chat_response = event

        return result
