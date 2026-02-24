import json
import uuid
from datetime import datetime
from typing import Literal, TypeAlias

from fastapi import APIRouter, HTTPException, Request

from server.middleware.auth import get_user_id
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
from models.openai.chat_completion_response_message import (
    ChatCompletionResponseMessage,
)
from models.openai.completion_usage import CompletionUsage
from models.openai.create_chat_completion_request import CreateChatCompletionRequest
from models.openai.create_chat_completion_response import (
    ChoicesItem,
    CreateChatCompletionResponse,
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
from models.message import Message, MessageRole, MessageContent, MessageContentType
from models.chat_response import ChatResponse
from utils.logging import llmmllogger
from utils import extract_text_from_message  # Import logging utility

import composer

OAIFinishReason: TypeAlias = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call"
]

logger = llmmllogger.bind(component="openai_chat_router")
router = APIRouter(prefix="/chat", tags=["Chat"])


def messages_from_openai(
    openai_messages: list[ChatCompletionRequestMessage],
) -> list[Message]:
    """Convert OpenAI chat completion request messages to internal Message format."""
    messages = []
    for oaim in openai_messages:
        contents = []
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
                            else MessageRole.USER  # Default to USER role if type is unrecognized
                        )
                    )
                )
            ),
            content=contents,
        )

        messages.append(msg)
    return messages


def openai_response_from_chat_response(
    chat_response: ChatResponse,
    model: str = "unknown",
) -> CreateChatCompletionResponse:
    """Convert internal ChatResponse to OpenAI CreateChatCompletionResponse format."""

    # Extract text content from the message
    content: str | None = None
    if chat_response.message and chat_response.message.content:
        text_parts = [
            part.text
            for part in chat_response.message.content
            if part.type == MessageContentType.TEXT and part.text
        ]
        content = "".join(text_parts) if text_parts else None

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
        oai_tool_calls = [
            ChatCompletionMessageToolCall(
                id=tc.execution_id or uuid.uuid4().hex,
                type="function",
                function=Function(
                    name=tc.name,
                    arguments=json.dumps(tc.args),
                ),
            )
            for tc in chat_response.message.tool_calls
        ]

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


@router.get("/completions")
async def listChatCompletions() -> ChatCompletionList:
    """Operation ID: listChatCompletions"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/completions")
async def createChatCompletion(
    body: CreateChatCompletionRequest,
    request: Request,
) -> CreateChatCompletionResponse:
    """Operation ID: createChatCompletion"""
    logger.info(json.dumps(body.model_dump(), indent=2))
    user_id = get_user_id(request)

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in request")

    builder = await composer.get_graph_builder(composer.WorkFlowType.IDE, user_id)
    workflow = await composer.compose_workflow(user_id, builder, None)
    initial_state = await builder.create_initial_state(
        user_id,
        0,
        messages_from_openai(body.messages),
    )
    chat_response: ChatResponse | None = None
    async for event in composer.execute_workflow(initial_state, workflow):
        print(
            extract_text_from_message(event.message) if event.message else "",
            flush=True,
            end="",
        )
        if event.finish_reason == "complete" and event.message:
            chat_response = event
    assert chat_response is not None, "Workflow did not produce a chat response"
    return openai_response_from_chat_response(chat_response)


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
