import json
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Union, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from server.middleware.auth import get_user_id
from models.anthropic.create_message_request import CreateMessageRequest
from models.anthropic.message_response import MessageResponse
from models.anthropic.count_tokens_request import CountTokensRequest
from models.anthropic.count_tokens_response import CountTokensResponse
from models.anthropic.output_content_block import OutputContentBlock
from models.anthropic.text_content_block import TextContentBlock
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
from utils.logging import llmmllogger


logger = llmmllogger.bind(component="anthropic_messages_router")
router = APIRouter(prefix="/messages", tags=["Messages"])


def messages_from_anthropic(
    anthropic_messages: list,
) -> list[Message]:
    """Convert Anthropic messages to internal Message format."""
    messages = []
    for msg in anthropic_messages:
        contents = []
        tool_calls = None

        # Handle content that can be either a string or a list of content blocks
        content = msg.content
        if isinstance(content, str):
            contents.append(MessageContent(type=MessageContentType.TEXT, text=content))
        elif isinstance(content, list):
            for block in content:
                if block.type == "text":
                    contents.append(
                        MessageContent(type=MessageContentType.TEXT, text=block.text)
                    )
                elif block.type == "tool_use":
                    # Convert tool_use blocks to tool_calls for internal format
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append(
                        ToolCall(
                            execution_id=block.id, name=block.name, args=block.input
                        )
                    )
                # Note: tool_result and other block types may need handling
                # For now, we focus on text and tool_use

        msg_role = MessageRole.USER if msg.role == "user" else MessageRole.ASSISTANT

        messages.append(Message(role=msg_role, content=contents, tool_calls=tool_calls))

    return messages


def anthropic_response_from_chat_response(
    chat_response: ChatResponse,
    model: str = "unknown",
    stop_reason: str = "end_turn",
) -> MessageResponse:
    """Convert internal ChatResponse to Anthropic MessageResponse format."""

    content_blocks: list[OutputContentBlock] = []

    if chat_response.message and chat_response.message.content:
        # Extract text content
        for part in chat_response.message.content:
            if part.type == MessageContentType.TEXT and part.text:
                content_blocks.append(TextContentBlock(type="text", text=part.text))

    # Extract tool calls and convert to tool_use blocks
    if chat_response.message and chat_response.message.tool_calls:
        for tc in chat_response.message.tool_calls:
            content_blocks.append(
                ToolUseContentBlock(
                    type="tool_use",
                    id=tc.execution_id or uuid.uuid4().hex,
                    name=tc.name,
                    input=tc.args,
                )
            )

    # Handle thinking content if present
    if chat_response.message and chat_response.message.thoughts:
        for thought in chat_response.message.thoughts:
            content_blocks.append(
                ThinkingContentBlock(
                    type="thinking", thinking=thought.text if thought.text else ""
                )
            )

    # Build usage
    usage = Usage(
        input_tokens=int(chat_response.prompt_eval_count or 0),
        output_tokens=int(chat_response.eval_count or 0),
    )

    # Map string stop_reason to literal type
    valid_stop_reasons: list[str] = [
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
    """Stream composer events as Anthropic SSE message chunks."""
    builder = await get_graph_builder(WorkFlowType.IDE, user_id)
    workflow = await compose_workflow(
        user_id=user_id,
        builder=builder,
        model_name=model_name,
        client_tools=client_tools,
        tool_choice=tool_choice,
    )
    initial_state = await create_initial_state(user_id, 0, builder)

    # For Anthropic streaming, we send chunks as server-sent events
    # Each chunk is a JSON object with event type and data

    has_tool_calls = False
    has_content = False
    final_tool_calls: list[ToolCall] = []
    final_content_blocks: list[OutputContentBlock] = []

    async for event in execute_workflow(initial_state, workflow):
        # Final accumulated event - capture tool calls and content
        if event.done:
            if event.message and event.message.tool_calls:
                final_tool_calls = event.message.tool_calls
                has_tool_calls = True
            if event.message and event.message.content:
                for part in event.message.content:
                    if part.type == MessageContentType.TEXT and part.text:
                        final_content_blocks.append(
                            TextContentBlock(type="text", text=part.text)
                        )
                    elif part.type == MessageContentType.TOOL_CALL:
                        # This shouldn't happen in content, but handle if present
                        pass
            continue

        # Stream text content deltas directly
        if event.message and event.message.content:
            for part in event.message.content:
                if part.type == MessageContentType.TEXT and part.text:
                    has_content = True
                    chunk = {
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": part.text},
                        "index": 0,
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

    # Send final content blocks if no streaming occurred
    if final_content_blocks and not has_content:
        for i, block in enumerate(final_content_blocks):
            # Determine delta type based on block type
            delta_type = "text_delta"
            delta_text = ""
            if block.type == "text" and hasattr(block, "text"):
                delta_type = "text_delta"
                delta_text = block.text
            elif block.type == "thinking" and hasattr(block, "thinking"):
                delta_type = "thinking_delta"
                delta_text = block.thinking
            elif block.type == "tool_use" and hasattr(block, "name"):
                delta_type = "tool_use_delta"
                delta_text = json.dumps(
                    {"name": block.name, "input": getattr(block, "input", {})}
                )

            chunk = {
                "type": "content_block_delta",
                "delta": {"type": delta_type, "text": delta_text},
                "index": i,
            }
            yield f"data: {json.dumps(chunk)}\n\n"

    # Send tool use blocks
    if final_tool_calls:
        for i, tc in enumerate(final_tool_calls):
            chunk = {
                "type": "content_block_delta",
                "delta": {
                    "type": "tool_use_delta",
                    "name": tc.name,
                    "input": json.dumps(tc.args),
                },
                "index": i + len(final_content_blocks),
            }
            yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk with stop reason
    stop_reason = "tool_use" if has_tool_calls else "end_turn"
    final_chunk = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason},
        "usage": {
            "output_tokens": sum(1 for _ in final_content_blocks)
            + sum(1 for _ in final_tool_calls)
        },
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/", response_model=None)
async def createMessage(
    body: CreateMessageRequest,
    request: Request,
) -> Union[MessageResponse, StreamingResponse]:
    """Operation ID: createMessage"""
    user_id = get_user_id(request)

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in request")

    logger.debug(body.model_dump_json(indent=2))
    # Convert Anthropic messages to internal format
    internal_messages = messages_from_anthropic(body.messages)

    logger.debug(body.model_dump_json(indent=2))

    # Convert Anthropic tool definitions to LangChain tools for bind_tools()
    client_tools = None
    tool_choice = None
    if body.tools:
        # Convert tool schemas to dict format for bind_tools()
        client_tools = []
        for tool in body.tools:
            client_tools.append(tool.model_dump(exclude_none=True))
        if body.tool_choice:
            # Map Anthropic tool_choice to OpenAI format
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
                "client_tools_created": len(client_tools) if client_tools else 0,
                "tool_choice": tool_choice,
            },
        )
    else:
        logger.debug("Anthropic request without tools")

    if body.stream:
        # Streaming response
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
    initial_state = await create_initial_state(user_id, 0, builder)
    chat_response: ChatResponse | None = None
    async for event in execute_workflow(initial_state, workflow):
        if event.finish_reason == "complete" and event.message:
            chat_response = event
    if chat_response is None:
        raise HTTPException(
            status_code=500, detail="Workflow did not produce a response"
        )

    # Determine stop_reason based on finish_reason
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


@router.post("/count_tokens")
async def countTokens(
    request: Request,
    body: CountTokensRequest,
) -> CountTokensResponse:
    """Operation ID: countTokens"""
    user_id = get_user_id(request)

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in request")

    raise NotImplementedError("Endpoint not yet implemented")
