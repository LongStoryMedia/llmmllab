import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Literal, TypeAlias, Union

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from middleware.auth import get_user_id
from services import CompletionService
from graph.state import ServerToolEvent
from models.openai.chat_completion_deleted import ChatCompletionDeleted
from models.openai.chat_completion_list import ChatCompletionList
from models.openai.chat_completion_message_list import ChatCompletionMessageList
from models.openai.chat_completion_message_custom_tool_call import (
    ChatCompletionMessageCustomToolCall,
)
from models.openai.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from models.openai.chat_completion_message_tool_calls import (
    ChatCompletionMessageToolCalls,
)
from models.openai.chat_completion_message_tool_call_chunk import (
    ChatCompletionMessageToolCallChunk,
    Function as ChunkFunction,
)
from models.openai.chat_completion_response_message import (
    ChatCompletionResponseMessage,
)
from models.openai.chat_completion_stream_response_delta import (
    ChatCompletionStreamResponseDelta,
)
from models.openai.completion_usage import CompletionUsage
from models.openai.create_chat_completion_request import CreateChatCompletionRequest
from models.openai.create_chat_completion_response import (
    ChoicesItem,
    CreateChatCompletionResponse,
)
from models.openai.create_chat_completion_stream_response import (
    ChoicesItem as StreamChoicesItem,
    CreateChatCompletionStreamResponse,
)
from models.openai.chat_completion_request_message import (
    ChatCompletionRequestMessage,
    ChatCompletionRequestMessageContentPartAudio,
    ChatCompletionRequestMessageContentPartFile,
    ChatCompletionRequestMessageContentPartRefusal,
    ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestMessageContentPartImage,
    ChatCompletionRequestToolMessage,
    ChatCompletionRequestUserMessage,
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestFunctionMessage,
    ChatCompletionRequestDeveloperMessage,
    ChatCompletionRequestSystemMessage,
)
from models.openai.chat_completion_tool import ChatCompletionTool
from models.message import Message, MessageRole, MessageContent, MessageContentType
from models.tool_call import ToolCall
from models.chat_response import ChatResponse
from utils.logging import llmmllogger

OAIFinishReason: TypeAlias = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call"
]

logger = llmmllogger.bind(component="openai_chat_router")
router = APIRouter(prefix="/chat", tags=["Chat"])


def openai_tools_as_dicts(openai_tools: list) -> list[dict]:
    """Convert ChatCompletionTool models to plain dicts for bind_tools().

    Passes the original OpenAI tool schemas through without lossy conversion.
    ChatOpenAI.bind_tools() accepts dicts in OpenAI format directly, which
    avoids the round-trip through Pydantic models that used to drop enum,
    anyOf, nested objects, and other JSON Schema features.
    """
    tools: list[dict] = []
    for tool_def in openai_tools:
        if not isinstance(tool_def, ChatCompletionTool):
            continue
        if tool_def.type != "function":
            continue
        tools.append(tool_def.model_dump(exclude_none=True))
    return tools


def messages_from_openai(
    openai_messages: list[ChatCompletionRequestMessage],
) -> list[Message]:
    """Convert OpenAI chat completion request messages to internal Message format."""
    messages = []
    for oaim in openai_messages:
        contents = []
        tool_call_id = None

        if isinstance(oaim.content, str):
            contents.append(
                MessageContent(type=MessageContentType.TEXT, text=oaim.content)
            )
        elif isinstance(oaim.content, list):
            for part in oaim.content:
                if isinstance(part, ChatCompletionRequestMessageContentPartText):
                    contents.append(
                        MessageContent(type=MessageContentType.TEXT, text=part.text)
                    )
                elif isinstance(part, ChatCompletionRequestMessageContentPartImage):
                    contents.append(
                        MessageContent(
                            type=MessageContentType.IMAGE,
                            url=part.image_url.url.encoded_string(),
                        )
                    )
                elif isinstance(part, ChatCompletionRequestMessageContentPartAudio):
                    contents.append(
                        MessageContent(
                            type=MessageContentType.AUDIO, url=part.input_audio.data
                        )
                    )
                elif isinstance(part, ChatCompletionRequestMessageContentPartFile):
                    contents.append(
                        MessageContent(
                            type=MessageContentType.FILE,
                            text=part.file.file_data,
                            name=part.file.filename,
                        )
                    )
                elif isinstance(part, ChatCompletionRequestMessageContentPartRefusal):
                    contents.append(
                        MessageContent(type=MessageContentType.TEXT, text=part.refusal)
                    )
                else:
                    logger.warning(
                        f"Unknown content part type: {type(part)}. Skipping."
                    )

        # Preserve tool_call_id for tool result messages via ToolCall
        tool_calls = None
        if isinstance(oaim, ChatCompletionRequestToolMessage):
            tool_call_id = oaim.tool_call_id
            tool_calls = [
                ToolCall(
                    name="tool_result",
                    execution_id=tool_call_id,
                    args={},
                )
            ]

        # Preserve tool_calls on assistant messages so LangChain can
        # pair AIMessage.tool_calls with subsequent ToolMessage entries.
        # Without this, the model never sees its own prior tool call
        # history and Copilot's multi-turn tool flow breaks.
        if isinstance(oaim, ChatCompletionRequestAssistantMessage) and oaim.tool_calls:
            # ChatCompletionMessageToolCalls is a RootModel; access .root for the list
            tc_list = (
                oaim.tool_calls.root
                if hasattr(oaim.tool_calls, "root")
                else oaim.tool_calls
            )
            tool_calls = []
            for tc in tc_list:
                if not isinstance(tc, ChatCompletionMessageToolCall):
                    continue
                try:
                    args = (
                        json.loads(tc.function.arguments)
                        if tc.function.arguments
                        else {}
                    )
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls.append(
                    ToolCall(
                        name=tc.function.name,
                        execution_id=tc.id,
                        args=args,
                    )
                )

        msg = Message(
            role=(
                MessageRole.USER
                if isinstance(oaim, ChatCompletionRequestUserMessage)
                or isinstance(oaim, ChatCompletionRequestDeveloperMessage)
                else (
                    MessageRole.ASSISTANT
                    if isinstance(oaim, ChatCompletionRequestAssistantMessage)
                    else (
                        MessageRole.SYSTEM
                        if isinstance(oaim, ChatCompletionRequestSystemMessage)
                        else (
                            MessageRole.TOOL
                            if isinstance(oaim, ChatCompletionRequestFunctionMessage)
                            or isinstance(oaim, ChatCompletionRequestToolMessage)
                            else MessageRole.USER
                        )
                    )
                )
            ),
            content=contents,
            tool_calls=tool_calls,
        )

        messages.append(msg)

    # Log message conversion summary for debugging multi-turn tool flows
    role_summary = {}
    for m in messages:
        key = m.role.value
        if m.tool_calls:
            key += f"(tc={len(m.tool_calls)})"
        role_summary[key] = role_summary.get(key, 0) + 1
    logger.debug(
        "Converted OpenAI messages",
        extra={"count": len(messages), "roles": role_summary},
    )

    return messages


def normalize_response(content_parts: list[MessageContent]) -> str:
    """Normalize the response content from message content parts into a single string."""
    return "\n".join(
        part.text
        for part in content_parts
        if part.type == MessageContentType.TEXT and part.text
    )


def normalize_tool_calls(
    tool_calls: list[ToolCall],
) -> list[ChatCompletionMessageToolCall]:
    """Normalize tool calls into the OpenAI chat completion format."""
    return [
        ChatCompletionMessageToolCall(
            id=tc.execution_id or uuid.uuid4().hex,
            type="function",
            function=Function(
                name=tc.name,
                arguments=json.dumps(tc.args),
            ),
        )
        for tc in tool_calls
    ]


def openai_response_from_chat_response(
    chat_response: ChatResponse,
    model: str = "unknown",
) -> CreateChatCompletionResponse:
    """Convert internal ChatResponse to OpenAI CreateChatCompletionResponse format."""

    # Extract text content from the message
    content: str | None = None
    if chat_response.message and chat_response.message.content:
        # Normalize the response content
        normalized_content = normalize_response(chat_response.message.content)
        content = normalized_content

    # Map internal finish_reason to OpenAI finish_reason
    finish_reason_map: dict[str | None, OAIFinishReason] = {
        "stop": "stop",
        "complete": "stop",
        "length": "length",
        "tool_call": "tool_calls",
        "error": "stop",
        "timeout": "stop",
        "cancel": "stop",
    }
    finish_reason: OAIFinishReason = finish_reason_map.get(
        chat_response.finish_reason, "stop"
    )

    # Convert internal ToolCalls to OpenAI ChatCompletionMessageToolCall list
    oai_tool_calls: list[
        ChatCompletionMessageToolCall | ChatCompletionMessageCustomToolCall
    ] = []
    if chat_response.message and chat_response.message.tool_calls:
        # Normalize tool calls
        normalized_tool_calls = normalize_tool_calls(chat_response.message.tool_calls)
        oai_tool_calls.extend(normalized_tool_calls)

    message = ChatCompletionResponseMessage(
        role="assistant",
        content=content,
        refusal=None,
        tool_calls=(
            ChatCompletionMessageToolCalls(oai_tool_calls) if oai_tool_calls else None
        ),
    )

    choice = ChoicesItem(
        index=0,
        message=message,
        finish_reason=finish_reason,
        logprobs=None,
    )

    # Build usage from token counts
    prompt_tokens = int(chat_response.prompt_eval_count or 0)
    completion_tokens = int(chat_response.eval_count or 0)
    usage = CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )

    # Build timestamp
    created = (
        int(chat_response.created_at.timestamp())
        if chat_response.created_at
        else int(datetime.now().timestamp())
    )

    return CreateChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=created,
        model=model,
        choices=[choice],
        usage=usage,
    )


async def stream_chat_completion(
    user_id: str,
    messages: list[Message],
    model_name: str,
    client_tools: list[dict] | None = None,
    tool_choice: str | None = None,
) -> AsyncIterator[str]:
    """Stream composer events as OpenAI SSE chat completion chunks.

    Retry, continuation, and nudge logic is handled by CompletionService.
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created = int(datetime.now().timestamp())

    # Send initial chunk with role
    initial_chunk = CreateChatCompletionStreamResponse(
        id=chunk_id,
        object="chat.completion.chunk",
        created=created,
        model=model_name,
        choices=[
            StreamChoicesItem.model_construct(
                index=0,
                delta=ChatCompletionStreamResponseDelta(role="assistant", content=""),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {initial_chunk.model_dump_json(exclude_none=True)}\n\n"

    has_tool_calls = False
    has_content = False
    final_tool_calls: list[ToolCall] = []
    final_content: str = ""

    from services import StreamAccumulator

    acc = StreamAccumulator()

    try:
        async for event, acc in CompletionService.stream_completion(
            user_id=user_id,
            messages=messages,
            model_name=model_name,
            client_tools=client_tools,
            tool_choice=tool_choice,
        ):
            # Skip ServerToolEvents for OpenAI-compatible clients

            if isinstance(event, ServerToolEvent):
                continue

            if event.done:
                if event.message and event.message.tool_calls:
                    final_tool_calls = event.message.tool_calls
                    has_tool_calls = True
                if event.message and event.message.content:
                    parts = [
                        c.text
                        for c in event.message.content
                        if c.type == MessageContentType.TEXT and c.text
                    ]
                    final_content = "".join(parts)
                continue

            # Stream text content deltas directly (skip thinking blocks)
            if event.message and event.message.content:
                text_parts = [
                    c.text
                    for c in event.message.content
                    if c.type == MessageContentType.TEXT and c.text
                ]
                response_text = "".join(text_parts)
                if response_text:
                    has_content = True
                    chunk = CreateChatCompletionStreamResponse(
                        id=chunk_id,
                        object="chat.completion.chunk",
                        created=created,
                        model=model_name,
                        choices=[
                            StreamChoicesItem.model_construct(
                                index=0,
                                delta=ChatCompletionStreamResponseDelta(
                                    content=response_text
                                ),
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    except asyncio.CancelledError:
        logger.warning("Client disconnected — stream_chat_completion cancelled")
        return

    # Use accumulator for final tool calls if not captured from done events
    if not has_tool_calls and acc.has_tool_calls:
        final_tool_calls = acc.final_tool_calls
        has_tool_calls = True
    if not final_content and acc.final_content:
        final_content = acc.final_content

    # Fallback: emit content if nothing was streamed
    if not has_content and not has_tool_calls and final_content:
        chunk = CreateChatCompletionStreamResponse(
            id=chunk_id,
            object="chat.completion.chunk",
            created=created,
            model=model_name,
            choices=[
                StreamChoicesItem.model_construct(
                    index=0,
                    delta=ChatCompletionStreamResponseDelta(content=final_content),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    # Stream tool calls from the final accumulated event
    if final_tool_calls:
        tool_call_chunks = []
        for i, tc in enumerate(final_tool_calls):
            tool_call_chunks.append(
                ChatCompletionMessageToolCallChunk(
                    index=i,
                    id=tc.execution_id or uuid.uuid4().hex,
                    type="function",
                    function=ChunkFunction(
                        name=tc.name,
                        arguments=json.dumps(tc.args),
                    ),
                )
            )
        chunk = CreateChatCompletionStreamResponse(
            id=chunk_id,
            object="chat.completion.chunk",
            created=created,
            model=model_name,
            choices=[
                StreamChoicesItem.model_construct(
                    index=0,
                    delta=ChatCompletionStreamResponseDelta(
                        tool_calls=tool_call_chunks,
                    ),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    # Final chunk with finish_reason
    finish_reason: OAIFinishReason = "tool_calls" if has_tool_calls else "stop"
    final_chunk = CreateChatCompletionStreamResponse(
        id=chunk_id,
        object="chat.completion.chunk",
        created=created,
        model=model_name,
        choices=[
            StreamChoicesItem.model_construct(
                index=0,
                delta=ChatCompletionStreamResponseDelta(),
                finish_reason=finish_reason,
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
    yield "data: [DONE]\n\n"


@router.get("/completions")
async def listChatCompletions() -> ChatCompletionList:
    """Operation ID: listChatCompletions"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/completions", response_model=None)
async def createChatCompletion(
    body: CreateChatCompletionRequest,
    request: Request,
) -> Union[CreateChatCompletionResponse, StreamingResponse]:
    """Operation ID: createChatCompletion"""
    user_id = get_user_id(request)

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in request")

    internal_messages = messages_from_openai(body.messages)

    # Convert OpenAI tool definitions to LangChain tools for bind_tools()
    client_tools = None
    tool_choice = None
    if body.tools:
        client_tools = openai_tools_as_dicts(body.tools)
        if body.tool_choice and isinstance(body.tool_choice, str):
            tool_choice = body.tool_choice
        logger.info(
            "OAI request with tools",
            extra={
                "tool_count": len(body.tools),
                "tool_names": [
                    t.function.name
                    for t in body.tools
                    if isinstance(t, ChatCompletionTool)
                ],
                "client_tools_created": len(client_tools) if client_tools else 0,
                "tool_choice": tool_choice,
            },
        )
    else:
        logger.debug("OAI request without tools")

    if body.stream:
        # Only pass tool kwargs when they have actual values to avoid
        # bypassing workflow caching with empty build_kwargs
        stream_kwargs: dict = {}
        if client_tools:
            stream_kwargs["client_tools"] = client_tools
        if tool_choice:
            stream_kwargs["tool_choice"] = tool_choice

        return StreamingResponse(
            stream_chat_completion(
                user_id,
                internal_messages,
                body.model,
                **stream_kwargs,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming response — delegate to CompletionService
    result = await CompletionService.run_completion(
        user_id=user_id,
        messages=internal_messages,
        model_name=body.model,
        client_tools=client_tools,
        tool_choice=tool_choice,
    )
    if result.chat_response is None:
        raise HTTPException(
            status_code=500, detail="Workflow did not produce a response"
        )
    return openai_response_from_chat_response(result.chat_response, model=body.model)


@router.delete("/completions/{completion_id}")
async def deleteChatCompletion(completion_id: str) -> ChatCompletionDeleted:
    """Operation ID: deleteChatCompletion"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/completions/{completion_id}")
async def getChatCompletion(completion_id: str) -> CreateChatCompletionResponse:
    """Operation ID: getChatCompletion"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/completions/{completion_id}")
async def updateChatCompletion(completion_id: str) -> CreateChatCompletionResponse:
    """Operation ID: updateChatCompletion"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/completions/{completion_id}/messages")
async def getChatCompletionMessages(completion_id: str) -> ChatCompletionMessageList:
    """Operation ID: getChatCompletionMessages"""
    raise NotImplementedError("Endpoint not yet implemented")
