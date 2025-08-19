"""
Utility functions for chat generation with streaming and non-streaming responses.
"""

import json
from typing import List, Optional, AsyncIterable, Union, Any
from datetime import datetime as dt

from models.chat_response import ChatResponse
from models.message import Message
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from models.message_role import MessageRole
from models.model_profile import ModelProfile
from server.context.conversation import ConversationContext
from server.config import logger
from fastapi.responses import StreamingResponse
from fastapi import BackgroundTasks

from server.utils.chat.message_utils import ensure_valid_message
from server.utils.chat.response_utils import create_valid_chat_response


async def generate_streaming_response(
    pipeline: Any,
    messages: List[Message],
    conversation_id: int,
    model_profile: Optional[ModelProfile],
    conversation_ctx: ConversationContext,
    background_tasks: Optional[BackgroundTasks] = None,
) -> AsyncIterable[str]:
    """
    Generate a streaming response.

    Args:
        pipeline: The model pipeline to use
        messages: The enhanced messages
        conversation_id: The conversation ID
        model_profile: The full model profile
        conversation_ctx: The conversation context
        background_tasks: FastAPI background tasks for async operations

    Yields:
        Streaming response chunks
    """
    # Use model profile to access parameters, system prompt, thinking options, and image settings
    options = (
        model_profile.parameters if model_profile and model_profile.parameters else {}
    )

    full_response = ""
    previous_message_id = None
    message_in_progress = False

    try:
        # Initialize message for streaming
        message_id = None
        start_time = dt.now()

        # Stream the response
        async for chunk in pipeline.stream(messages, options):
            # Accumulate the full response
            full_response += chunk

            # Create a message content object for this chunk
            content = MessageContent(type=MessageContentType.TEXT, text=chunk, url=None)

            # Create the message for this chunk
            message_data = {
                "role": MessageRole.ASSISTANT,
                "content": [content],
                "conversation_id": conversation_id,
                "tool_calls": None,
                "thinking": (
                    str(model_profile.parameters.think)
                    if model_profile
                    and model_profile.parameters
                    and model_profile.parameters.think
                    else None
                ),
                "id": message_id,  # Use the same ID for the entire stream
                "created_at": dt.now(),
            }

            # Ensure the message has valid content and conversation_id
            partial_message = ensure_valid_message(message_data, conversation_id)

            # Only set message ID on the first chunk to maintain consistency
            if not message_id:
                message_id = partial_message.id
            else:
                partial_message.id = message_id

            # Create a validated response using our helper function
            streaming_response = create_valid_chat_response(
                done=False,
                message=partial_message,
                created_at=dt.now(),
                model=model_profile.model_name if model_profile else "default_model",
            )

            # Send the streaming chunk
            yield f"data: {json.dumps(streaming_response.dict())}\n\n"

        # Create the final message with the complete response
        final_content = MessageContent(
            type=MessageContentType.TEXT, text=full_response, url=None
        )

        # Create the message with the final content
        message_data = {
            "role": MessageRole.ASSISTANT,
            "content": [final_content],
            "conversation_id": conversation_id,
            "tool_calls": None,
            "thinking": (
                str(model_profile.parameters.think)
                if model_profile
                and model_profile.parameters
                and model_profile.parameters.think
                else None
            ),
            "id": message_id,  # Use the same ID for consistency
            "created_at": dt.now(),
        }

        # Ensure final message has valid content and conversation_id
        final_message = ensure_valid_message(message_data, conversation_id)

        # Create a validated response using our helper function
        final_response = create_valid_chat_response(
            done=True,
            message=final_message,
            created_at=dt.now(),
            model=model_profile.model_name if model_profile else "default_model",
            finish_reason="stop",
        )

        yield f"data: {json.dumps(final_response.dict())}\n\n"

        # Store the final message in the background
        if final_message and background_tasks is not None:
            background_tasks.add_task(
                store_assistant_message, conversation_ctx, final_message
            )

    except Exception as e:
        logger.error(f"Error in streaming generation: {e}")
        # Handle error by sending a final error message
        error_message = Message(
            role=MessageRole.ASSISTANT,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text=f"An error occurred during response generation: {str(e)}",
                    url=None,
                )
            ],
            conversation_id=conversation_id,
        )

        error_response = create_valid_chat_response(
            done=True,
            message=error_message,
            created_at=dt.now(),
            model="error",
            finish_reason="error",
        )

        yield f"data: {json.dumps(error_response.dict())}\n\n"


async def generate_complete_response(
    pipeline: Any,
    messages: List[Message],
    conversation_id: int,
    model_profile: Optional[ModelProfile],
    conversation_ctx: ConversationContext,
) -> ChatResponse:
    """Generate a complete non-streaming response."""
    # Generate the complete response
    # Use model profile to access parameters, system prompt, thinking options, and image settings
    options = (
        model_profile.parameters if model_profile and model_profile.parameters else {}
    )

    # Pass options to the pipeline
    response_content = await pipeline.generate(messages, options)

    # Ensure response content is valid
    if not response_content:
        response_content = "I'm sorry, I couldn't generate a response."

    # Always create a proper MessageContent object
    content_item = MessageContent(
        type=MessageContentType.TEXT, text=response_content, url=None
    )

    # Create the message and ensure it's valid
    message_data = {
        "role": MessageRole.ASSISTANT,
        "content": [content_item],
        "conversation_id": conversation_id,
        "tool_calls": None,
        "thinking": (
            str(model_profile.parameters.think)
            if model_profile
            and model_profile.parameters
            and model_profile.parameters.think
            else None
        ),
        "id": None,
        "created_at": dt.now(),
    }

    # Validate the message
    response_message = ensure_valid_message(message_data, conversation_id)

    # Create and return a validated ChatResponse
    return create_valid_chat_response(
        done=True,
        message=response_message,
        created_at=dt.now(),
        model=model_profile.model_name if model_profile else "default_model",
        finish_reason="stop",
    )


async def store_assistant_message(
    conversation_ctx: ConversationContext, message: Message
):
    """Store the assistant message in the conversation context."""
    try:
        if conversation_ctx and hasattr(conversation_ctx, "add_assistant_message"):
            await conversation_ctx.add_assistant_message(message)
        else:
            logger.warning(
                "Unable to store assistant message: invalid conversation context"
            )
    except Exception as e:
        logger.error(f"Failed to store assistant message: {e}")


async def stream_agentic_response(
    response_text: str,
    conversation_id: int,
    model_profile: Optional[ModelProfile],
    conversation_ctx: ConversationContext,
    background_tasks: Optional[BackgroundTasks],
):
    """
    Stream an agentic response that was generated non-streaming
    """
    # Split response into chunks for streaming effect
    words = response_text.split()
    chunk_size = 3  # Words per chunk
    message_id = None

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])

        # Create message content for this chunk
        content = MessageContent(type=MessageContentType.TEXT, text=chunk, url=None)

        # Create message object
        message_data = {
            "role": MessageRole.ASSISTANT,
            "content": [content],
            "conversation_id": conversation_id,
            "id": message_id,  # Use the same ID throughout streaming
            "created_at": dt.now(),
        }

        # Validate the message
        partial_message = ensure_valid_message(message_data, conversation_id)

        # Set message ID on first chunk
        if not message_id:
            message_id = partial_message.id
        else:
            partial_message.id = message_id

        # Create response
        streaming_response = create_valid_chat_response(
            done=False,
            message=partial_message,
            model=model_profile.model_name if model_profile else "default_model",
        )

        yield f"data: {json.dumps(streaming_response.dict())}\n\n"

    # Send final done message
    final_content = MessageContent(
        type=MessageContentType.TEXT, text=response_text, url=None
    )

    # Create the message and ensure it's valid
    message_data = {
        "role": MessageRole.ASSISTANT,
        "content": [final_content],
        "conversation_id": conversation_id,
        "tool_calls": None,
        "thinking": (
            str(model_profile.parameters.think)
            if model_profile
            and model_profile.parameters
            and model_profile.parameters.think
            else None
        ),
        "id": message_id,
        "created_at": dt.now(),
    }
    # Ensure final message has valid content and conversation_id
    final_message = ensure_valid_message(message_data, conversation_id)

    # Create a validated response using our helper function
    final_response = create_valid_chat_response(
        done=True,
        message=final_message,
        created_at=dt.now(),
        model=model_profile.model_name if model_profile else "default_model",
        finish_reason="stop",
    )

    yield f"data: {json.dumps(final_response.dict())}\n\n"

    # Store the final message
    if final_message and background_tasks is not None:
        background_tasks.add_task(
            store_assistant_message, conversation_ctx, final_message
        )
