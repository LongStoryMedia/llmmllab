import json
import uuid
from collections.abc import AsyncIterator
from typing import Dict, Union, Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
import regex

from server.middleware.auth import get_user_id
from models.anthropic.create_message_request import CreateMessageRequest
from models.anthropic.message_response import MessageResponse
from models.anthropic.count_tokens_request import CountTokensRequest
from models.anthropic.count_tokens_response import CountTokensResponse
from models.anthropic.output_content_block import OutputContentBlock
from models.anthropic.text_content_block import TextContentBlock
from models.anthropic.tool_reference_content_block import ToolReferenceContentBlock
from models.anthropic.tool_result_content_block import ToolResultContentBlock
from models.anthropic.tool_use_content_block import ToolUseContentBlock
from models.anthropic.thinking_content_block import ThinkingContentBlock
from models.anthropic.usage import Usage
from models.message import Message, MessageRole, MessageContent, MessageContentType
from models.tool_call import ToolCall
from models.chat_response import ChatResponse
from composer import (
    compose_workflow,
    create_initial_state,
    execute_workflow,
    get_graph_builder,
)
from composer.graph.workflows.factory import WorkFlowType
from runner import pipeline_factory
from runner.pipelines.llamacpp.chat import ChatLlamaCppPipeline
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="anthropic_messages_router")
router = APIRouter(prefix="/messages", tags=["Messages"])

# Claude Code determines the context window from the model name and triggers
# auto-compaction at ~83.5% of that limit. All current Claude models have a
# 200K context window. Since we proxy to a local model with a smaller context,
# we scale reported token counts so that X% of our actual num_ctx appears as
# X% of 200K — making compaction fire at the right time.
_CLAUDE_ASSUMED_CONTEXT = 200_000

# Continuation retry constants: when the model produces text indicating intent
# to use a tool but emits EOS before the tool call, we retry with a nudge.
_CONTINUATION_PROMPT = (
    "You described an action you intended to take but did not call any tool. "
    "Please call the appropriate tool now to carry out that action."
)
_MAX_TOOL_CONTINUATIONS = 2


def _get_num_ctx() -> int:
    """Get num_ctx from the active pipeline's model profile."""
    try:
        cache = pipeline_factory.local_cache
        with cache._lock:
            for entry in cache._cache.values():
                pipeline = entry.pipeline
                if isinstance(pipeline, ChatLlamaCppPipeline) and hasattr(
                    pipeline, "profile"
                ):
                    num_ctx = pipeline.profile.parameters.num_ctx
                    if num_ctx:
                        return num_ctx
    except Exception:
        pass
    return 131_072  # safe default matching primary profile


def _scale_tokens(actual: int) -> int:
    """Scale actual token count to Claude's assumed 200K context window.

    We treat the effective context as 90% of num_ctx so that Claude Code's
    83.5% compaction threshold fires at ~75% of our real context limit
    (0.835 * 0.9 ≈ 0.75), leaving headroom for tool-heavy turns.
    """
    num_ctx = _get_num_ctx()
    effective_ctx = int(num_ctx * 0.90)
    if effective_ctx >= _CLAUDE_ASSUMED_CONTEXT:
        return actual
    return int(actual * _CLAUDE_ASSUMED_CONTEXT / effective_ctx)


async def _count_input_tokens(
    messages: list[Message],
    tools: list | None = None,
) -> int:
    """Count input tokens by calling llama-server /tokenize, with char-estimate fallback.

    Builds a plain-text representation of the conversation and tool definitions,
    then tokenizes via the running llama-server. Falls back to len // 4 estimate.
    """
    parts: list[str] = []
    for msg in messages:
        role_tag = msg.role.value if msg.role else "user"
        text = ""
        if msg.content:
            text = " ".join(
                c.text
                for c in msg.content
                if c.type == MessageContentType.TEXT and c.text
            )
        parts.append(f"<|{role_tag}|>\n{text}")

    if tools:
        for tool in tools:
            if isinstance(tool, dict):
                parts.append(json.dumps(tool))
            else:
                parts.append(json.dumps(tool.model_dump(exclude_none=True)))

    combined_text = "\n".join(parts)

    try:
        cache = pipeline_factory.local_cache
        with cache._lock:
            server_url = None
            for entry in cache._cache.values():
                pipeline = entry.pipeline
                if (
                    isinstance(pipeline, ChatLlamaCppPipeline)
                    and pipeline.server_manager
                ):
                    server_url = pipeline.server_manager.server_url
                    break

        if server_url:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{server_url}/tokenize",
                    json={"content": combined_text},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    tokens = data.get("tokens", [])
                    return len(tokens)
    except Exception as e:
        logger.debug(f"llama-server tokenize unavailable, using estimate: {e}")

    return max(1, len(combined_text) // 4)


def _sse(event_type: str, data: dict) -> str:
    """Format a server-sent event with the required Anthropic event/data structure."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def messages_from_anthropic(
    anthropic_messages: list,
    system: Any = None,
) -> list[Message]:
    """Convert Anthropic messages to internal Message format.

    Handles:
    - String content
    - Text and tool_use blocks in assistant messages
    - tool_result blocks in user messages (expanded to TOOL role messages)
    - System prompt (prepended as SYSTEM message)
    """
    messages: list[Message] = []

    # Prepend system message if present
    if system is not None:
        if isinstance(system, str):
            system_text = system
        else:
            # List of TextContentBlock
            system_text = "\n".join(
                block.text for block in system if hasattr(block, "text") and block.text
            )
        if system_text:
            messages.append(
                Message(
                    role=MessageRole.SYSTEM,
                    content=[
                        MessageContent(type=MessageContentType.TEXT, text=system_text)
                    ],
                )
            )

    for msg in anthropic_messages:
        content = msg.content

        # Simple string content
        if isinstance(content, str):
            role = MessageRole.USER if msg.role == "user" else MessageRole.ASSISTANT
            messages.append(
                Message(
                    role=role,
                    content=[
                        MessageContent(type=MessageContentType.TEXT, text=content)
                    ],
                )
            )
            continue

        # List of content blocks
        if msg.role == "user":
            tool_result_blocks = [
                b for b in content if hasattr(b, "type") and b.type == "tool_result"
            ]
            if tool_result_blocks:
                # Each tool_result block becomes a separate TOOL message (mirrors OAI tool messages)
                for block in tool_result_blocks:
                    result_text = ""
                    if isinstance(block.content, str):
                        result_text = block.content
                    elif isinstance(block.content, list):
                        # Handle mixed content: text, tool_reference, etc.
                        parts = []
                        for item in block.content:
                            if hasattr(item, "text") and item.text:
                                parts.append(item.text)
                            elif isinstance(item, ToolReferenceContentBlock):
                                # Format tool reference as readable text
                                parts.append(f"[Tool: {item.tool_name}]")
                        result_text = "\n".join(parts)
                    messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=[
                                MessageContent(
                                    type=MessageContentType.TEXT, text=result_text
                                )
                            ],
                            tool_calls=[
                                ToolCall(
                                    name="tool_result",
                                    execution_id=block.tool_use_id,
                                    args={},
                                )
                            ],
                        )
                    )
                # Handle any non-tool_result text blocks in the same user message
                other_text = [
                    b.text
                    for b in content
                    if hasattr(b, "type")
                    and b.type == "text"
                    and hasattr(b, "text")
                    and b.text
                ]
                if other_text:
                    messages.append(
                        Message(
                            role=MessageRole.USER,
                            content=[
                                MessageContent(
                                    type=MessageContentType.TEXT,
                                    text="\n".join(other_text),
                                )
                            ],
                        )
                    )
                continue

        # Regular user or assistant message with text and/or tool_use blocks
        text_contents: list[MessageContent] = []
        tool_calls: list[ToolCall] | None = None

        for block in content:
            if not hasattr(block, "type"):
                continue
            if block.type == "text":
                text_contents.append(
                    MessageContent(type=MessageContentType.TEXT, text=block.text)
                )
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    ToolCall(
                        execution_id=block.id,
                        name=block.name,
                        args=block.input if isinstance(block.input, dict) else {},
                    )
                )

        role = MessageRole.USER if msg.role == "user" else MessageRole.ASSISTANT
        if text_contents or tool_calls:
            messages.append(
                Message(
                    role=role,
                    content=text_contents
                    or [MessageContent(type=MessageContentType.TEXT, text="")],
                    tool_calls=tool_calls,
                )
            )

    return messages


def anthropic_response_from_chat_response(
    chat_response: ChatResponse,
    model: str = "unknown",
    stop_reason: str = "end_turn",
) -> MessageResponse:
    """Convert internal ChatResponse to Anthropic MessageResponse format."""

    content_blocks: list[OutputContentBlock] = []

    # Thinking blocks first (per Anthropic spec ordering)
    if chat_response.message and chat_response.message.thoughts:
        for thought in chat_response.message.thoughts:
            content_blocks.append(
                ThinkingContentBlock(
                    type="thinking", thinking=thought.text if thought.text else ""
                )
            )

    # Text blocks
    if chat_response.message and chat_response.message.content:
        for part in chat_response.message.content:
            if part.type == MessageContentType.TEXT and part.text:
                content_blocks.append(TextContentBlock(type="text", text=part.text))

    # Tool use blocks
    if chat_response.message and chat_response.message.tool_calls:
        for tc in chat_response.message.tool_calls:
            content_blocks.append(
                ToolUseContentBlock(
                    type="tool_use",
                    id=tc.execution_id or f"toolu_{uuid.uuid4().hex[:24]}",
                    name=tc.name,
                    input=tc.args,
                )
            )

    usage = Usage(
        input_tokens=_scale_tokens(int(chat_response.prompt_eval_count or 0)),
        output_tokens=int(chat_response.eval_count or 0),
    )

    valid_stop_reasons = [
        "end_turn",
        "max_tokens",
        "stop_sequence",
        "tool_use",
        "pause_turn",
    ]
    actual_stop_reason = (
        stop_reason if stop_reason in valid_stop_reasons else "end_turn"
    )

    return MessageResponse(
        id=f"msg_{uuid.uuid4().hex[:24]}",
        type="message",
        role="assistant",
        content=content_blocks,
        model=model,
        stop_reason=actual_stop_reason,  # type: ignore
        usage=usage,
    )


async def stream_message(
    user_id: str,
    messages: list[Message],
    model_name: str,
    client_tools: list | None = None,
    tool_choice: str | None = None,
) -> AsyncIterator[str]:
    """Stream composer events as Anthropic SSE message chunks.

    Emits the full Anthropic streaming event sequence:
      message_start → ping → content_block_start → content_block_delta(s)
      → content_block_stop → message_delta → message_stop

    If the model produces text indicating intent to use a tool but does not
    actually emit any tool_use blocks (common with smaller local models), a
    single retry is attempted with a continuation prompt to coax the tool call.
    """
    builder = await get_graph_builder(WorkFlowType.IDE, user_id)
    workflow = await compose_workflow(
        user_id=user_id,
        builder=builder,
        model_name=model_name,
        client_tools=client_tools,
        tool_choice=tool_choice,
    )
    initial_state = await create_initial_state(user_id, 0, builder, messages)

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Pre-compute input tokens so message_start reports accurate context usage.
    # Scale to Claude's 200K window so Claude Code triggers compaction at the
    # right percentage of our actual num_ctx.
    raw_input_tokens = await _count_input_tokens(messages, client_tools)
    input_tokens = _scale_tokens(raw_input_tokens)

    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model_name,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        },
    )
    yield _sse("ping", {"type": "ping"})

    text_block_started = False
    text_block_index = 0
    has_content = False
    has_tool_calls = False
    final_tool_calls: list[ToolCall] = []
    final_content: str = ""
    output_tokens = 0

    async for event in execute_workflow(initial_state, workflow):
        if event.done:
            logger.debug(
                "Received done event",
                extra={
                    "has_message": bool(event.message),
                    "has_tool_calls": bool(event.message and event.message.tool_calls),
                    "has_content": bool(event.message and event.message.content),
                },
            )
            if event.message and event.message.tool_calls:
                final_tool_calls = event.message.tool_calls
                has_tool_calls = True
            # Capture final content as fallback for non-streamed responses
            # (e.g. when thinking was promoted to content by the executor)
            if event.message and event.message.content:
                parts = [
                    c.text
                    for c in event.message.content
                    if c.type == MessageContentType.TEXT and c.text
                ]
                final_content = "".join(parts)
                logger.debug(
                    "Captured final content from done event",
                    extra={"content_len": len(final_content), "content_preview": final_content[:200]},
                )
            if event.prompt_eval_count:
                input_tokens = _scale_tokens(int(event.prompt_eval_count))
            if event.eval_count:
                output_tokens = int(event.eval_count)
            continue

        # Stream live text deltas
        if event.message and event.message.content:
            for part in event.message.content:
                if part.type == MessageContentType.TEXT and part.text:
                    logger.debug(
                        "Streaming content delta",
                        extra={"text_len": len(part.text), "text_preview": part.text[:100]},
                    )
                    if not text_block_started:
                        yield _sse(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": text_block_index,
                                "content_block": {"type": "text", "text": ""},
                            },
                        )
                        text_block_started = True
                    has_content = True
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": text_block_index,
                            "delta": {"type": "text_delta", "text": part.text},
                        },
                    )

    # ----------------------------------------------------------------
    # Continuation retry: if the model produced text but no tool calls
    # and tools were provided, nudge the model to actually call a tool.
    # This handles the common small-model failure mode of narrating an
    # action ("Let me read the file:") then emitting EOS before the
    # tool call.
    # ----------------------------------------------------------------
    continuations = 0
    while (
        not has_tool_calls
        and client_tools
        and (has_content or final_content)
        and continuations < _MAX_TOOL_CONTINUATIONS
    ):
        continuations += 1
        accumulated_text = final_content or ""
        logger.info(
            "Model produced text without tool calls — attempting continuation retry",
            extra={
                "attempt": continuations,
                "content_len": len(accumulated_text),
                "content_preview": accumulated_text[:200],
            },
        )

        # Build continuation messages: original messages + assistant text + nudge
        continuation_messages = list(messages) + [
            Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(type=MessageContentType.TEXT, text=accumulated_text)
                ],
            ),
            Message(
                role=MessageRole.USER,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=_CONTINUATION_PROMPT,
                    )
                ],
            ),
        ]

        retry_builder = await get_graph_builder(WorkFlowType.IDE, user_id)
        retry_workflow = await compose_workflow(
            user_id=user_id,
            builder=retry_builder,
            model_name=model_name,
            client_tools=client_tools,
            tool_choice="required",  # Force tool use on retry
        )
        retry_state = await create_initial_state(
            user_id, 0, retry_builder, continuation_messages
        )

        async for event in execute_workflow(retry_state, retry_workflow):
            if event.done:
                if event.message and event.message.tool_calls:
                    final_tool_calls = event.message.tool_calls
                    has_tool_calls = True
                # Capture any additional content from the retry
                if event.message and event.message.content:
                    parts = [
                        c.text
                        for c in event.message.content
                        if c.type == MessageContentType.TEXT and c.text
                    ]
                    retry_text = "".join(parts)
                    if retry_text:
                        final_content += retry_text
                if event.eval_count:
                    output_tokens += int(event.eval_count)
                continue

            # Stream additional text from the retry
            if event.message and event.message.content:
                for part in event.message.content:
                    if part.type == MessageContentType.TEXT and part.text:
                        if not text_block_started:
                            yield _sse(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": text_block_index,
                                    "content_block": {"type": "text", "text": ""},
                                },
                            )
                            text_block_started = True
                        has_content = True
                        yield _sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": text_block_index,
                                "delta": {"type": "text_delta", "text": part.text},
                            },
                        )

    # Fallback: emit content from done event if nothing was streamed live
    if not has_content and not has_tool_calls and final_content:
        if not text_block_started:
            yield _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": text_block_index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            text_block_started = True
        yield _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": text_block_index,
                "delta": {"type": "text_delta", "text": final_content},
            },
        )

    # Safety net: if model produced absolutely nothing (no content, no tool
    # calls), emit a minimal text block so the client doesn't receive a
    # completely empty assistant turn which causes Claude Code to silently
    # drop the response.
    if not has_content and not has_tool_calls and not final_content:
        logger.warning(
            "Model produced empty response — no content or tool calls",
            extra={"model": model_name, "input_tokens": input_tokens},
        )
        if not text_block_started:
            yield _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": text_block_index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            text_block_started = True
        yield _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": text_block_index,
                "delta": {
                    "type": "text_delta",
                    "text": "[Model returned empty response. The context may be too large or the model may need to be reloaded. Please try again.]",
                },
            },
        )

    # Close the text block
    if text_block_started:
        yield _sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": text_block_index},
        )

    # Emit tool_use blocks (always come from the final done event)
    tool_block_start = text_block_index + (1 if text_block_started else 0)
    for i, tc in enumerate(final_tool_calls):
        block_index = tool_block_start + i
        tc_id = tc.execution_id or f"toolu_{uuid.uuid4().hex[:24]}"

        yield _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": block_index,
                "content_block": {
                    "type": "tool_use",
                    "id": tc_id,
                    "name": tc.name,
                    "input": {},
                },
            },
        )
        yield _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(tc.args),
                },
            },
        )
        yield _sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": block_index},
        )

    stop_reason = "tool_use" if has_tool_calls else "end_turn"
    logger.debug(
        "Stream complete",
        extra={
            "has_content": has_content,
            "has_tool_calls": has_tool_calls,
            "final_content_len": len(final_content),
            "stop_reason": stop_reason,
            "text_block_started": text_block_started,
        },
    )
    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})


@router.post("", response_model=None)
async def createMessage(
    req_body: Dict[str, Any],
    request: Request,
) -> Union[MessageResponse, StreamingResponse]:
    """Operation ID: createMessage"""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in request")

    try:
        body = CreateMessageRequest.model_validate(req_body)
        internal_messages = messages_from_anthropic(body.messages, system=body.system)
        claude_regex = regex.compile(r"claude|haiku|sonnet|opus", regex.IGNORECASE)
        if claude_regex.search(body.model):
            body.model = "Qwen3_5_0_8B"

        client_tools = None
        tool_choice = None
        if body.tools:
            client_tools = [tool.model_dump(exclude_none=True) for tool in body.tools]
            if body.tool_choice:
                if body.tool_choice.type == "any":
                    tool_choice = "required"
                elif body.tool_choice.type == "auto":
                    tool_choice = "auto"
                elif body.tool_choice.type == "tool":
                    tool_choice = "tool"
            logger.info(
                "Anthropic request with tools",
                extra={
                    "tool_count": len(body.tools),
                    "tool_names": [t.name for t in body.tools],
                    "client_tools_created": len(client_tools),
                    "tool_choice": tool_choice,
                },
            )
        else:
            logger.debug(
                f"Anthropic request without tools: {body.model_dump_json(indent=2)}"
            )
            # body.stream = True

        if body.stream:
            return StreamingResponse(
                stream_message(
                    user_id,
                    internal_messages,
                    body.model,
                    client_tools=client_tools,
                    tool_choice=tool_choice,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming response
        builder = await get_graph_builder(WorkFlowType.IDE, user_id)
        workflow = await compose_workflow(
            user_id=user_id,
            builder=builder,
            model_name=body.model,
            client_tools=client_tools,
            tool_choice=tool_choice,
        )
        initial_state = await create_initial_state(
            user_id,
            0,
            builder,
            internal_messages,
        )
        chat_response: ChatResponse | None = None
        async for event in execute_workflow(initial_state, workflow):
            if event.done and event.message:
                chat_response = event
        if chat_response is None:
            raise HTTPException(
                status_code=500, detail="Workflow did not produce a response"
            )

        # Continuation retry for non-streaming: if the model produced text
        # but no tool calls and tools were provided, nudge it to call a tool.
        continuations = 0
        while (
            client_tools
            and chat_response is not None
            and chat_response.message
            and chat_response.message.content
            and not chat_response.message.tool_calls
            and continuations < _MAX_TOOL_CONTINUATIONS
        ):
            continuations += 1
            accumulated_text = "".join(
                c.text
                for c in chat_response.message.content
                if c.type == MessageContentType.TEXT and c.text
            )
            if not accumulated_text:
                break
            logger.info(
                "Non-streaming: model produced text without tool calls — attempting continuation retry",
                extra={
                    "attempt": continuations,
                    "content_len": len(accumulated_text),
                    "content_preview": accumulated_text[:200],
                },
            )
            continuation_messages = list(internal_messages) + [
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
                            type=MessageContentType.TEXT,
                            text=_CONTINUATION_PROMPT,
                        )
                    ],
                ),
            ]
            retry_builder = await get_graph_builder(WorkFlowType.IDE, user_id)
            retry_workflow = await compose_workflow(
                user_id=user_id,
                builder=retry_builder,
                model_name=body.model,
                client_tools=client_tools,
                tool_choice="required",
            )
            retry_state = await create_initial_state(
                user_id, 0, retry_builder, continuation_messages
            )
            async for event in execute_workflow(retry_state, retry_workflow):
                if event.done and event.message:
                    chat_response = event

        stop_reason_map: dict[str | None, str] = {
            "stop": "end_turn",
            "complete": "end_turn",
            "length": "max_tokens",
            "tool_call": "tool_use",
        }
        stop_reason = stop_reason_map.get(chat_response.finish_reason, "end_turn")

        return anthropic_response_from_chat_response(
            chat_response, model=body.model, stop_reason=stop_reason
        )
    except ValidationError as ve:
        logger.error(f"Validation error in createMessage request: {ve.json()}")
        raise HTTPException(status_code=422, detail=json.loads(ve.json()))

    except Exception as e:
        logger.error(f"Error processing createMessage request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/count_tokens")
async def countTokens(
    request: Request,
    body: CountTokensRequest,
) -> CountTokensResponse:
    """Operation ID: countTokens

    Estimates the token count for a message request by forwarding the
    rendered text to the running llama-server's /tokenize endpoint.
    """
    user_id = get_user_id(request)

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in request")

    try:
        internal_messages = messages_from_anthropic(body.messages, system=body.system)
        raw_count = await _count_input_tokens(internal_messages, body.tools)
        return CountTokensResponse(input_tokens=_scale_tokens(raw_count))

    except Exception as e:
        logger.error(f"Error in countTokens: {e}")
        raise HTTPException(status_code=400, detail=str(e))
