"""
Chat completion logic for generating responses, streaming, and handling agentic workflows.
"""

import asyncio
import json
from datetime import datetime as dt
from typing import List, Any, Optional, AsyncIterable, Union, Dict

from fastapi import BackgroundTasks, HTTPException, status
from fastapi.responses import StreamingResponse

from server.db import storage
from models.message import Message
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from models.message_role import MessageRole
from models.model_profile import ModelProfile
from models.chat_response import ChatResponse
from server.context.conversation import ConversationContext
from server.config import logger
from server.tools import create_agentic_chat_completion
from server.utils.chat import (
    create_valid_chat_response,
    should_use_agentic_workflow,
    ensure_valid_message,
    extract_message_text,
)
from runner.pipelines.factory import pipeline_factory

# Type definition for pipeline
PipelineProtocol = Any  # Using Any since Protocol definition was causing lint errors


async def enhanced_chat_completion_logic(
    conversation_ctx: ConversationContext,
    background_tasks: Optional[BackgroundTasks] = None,
) -> StreamingResponse:
    """
    Enhanced chat completion logic that can use agentic workflow or standard pipeline.
    Always streams responses for real-time interaction.

    Args:
        conversation_ctx: The conversation context containing messages
        background_tasks: Optional background tasks for async operations

    Returns:
        StreamingResponse containing the chat completion
    """
    # Get the conversation ID from context
    conversation_id = conversation_ctx.conversation_id

    # Get current user message
    user_message = conversation_ctx.current_user_message
    assert user_message, "Current user message not set"

    # Extract text from user message for intent analysis
    user_text = extract_message_text(user_message).strip()

    # Determine if we should use agentic workflow
    use_agentic = should_use_agentic_workflow(user_text)

    # Get the model profile from user config
    model_profile = await storage.get_service(
        storage.model_profile
    ).get_model_profile_by_id(
        conversation_ctx.user_config.model_profiles.primary_profile_id,
        conversation_ctx.user_id,
    )

    # Get enhanced messages from conversation context
    from server.utils.chat.workflow import prepare_enhanced_messages

    enhanced_messages = prepare_enhanced_messages(conversation_ctx, model_profile)

    # AGENTIC WORKFLOW PATH
    if use_agentic:
        logger.info(f"Using agentic workflow for conversation {conversation_id}")
        try:
            # Get appropriate engineering model profile for agentic processing
            mp = await storage.get_service(
                storage.model_profile
            ).get_model_profile_by_id(
                conversation_ctx.user_config.model_profiles.engineering_profile_id,
                conversation_ctx.user_id,
            )
            assert mp, "Engineering model profile not found"

            # Generate the response using agentic workflow
            response_text = await create_agentic_chat_completion(
                conversation_ctx=conversation_ctx,
                model_id=mp.name,
            )

            # Always stream the agentic response
            return StreamingResponse(
                stream_agentic_response(
                    response_text,
                    conversation_id,
                    model_profile,
                    conversation_ctx,
                    background_tasks,
                ),
                media_type="text/event-stream",
            )
        except Exception as e:
            logger.error(f"Error in agentic workflow: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error in agentic workflow: {str(e)}",
            )

    # STANDARD PIPELINE PATH
    # Get appropriate model ID
    model_id = str(model_profile.id) if model_profile else "default"

    # Get pipeline from factory
    pipeline_result, _ = pipeline_factory.get_pipeline(model_id)
    pipeline: PipelineProtocol = pipeline_result

    # Always stream the response
    return StreamingResponse(
        generate_streaming_response(
            pipeline,
            enhanced_messages,
            conversation_id,
            model_profile,
            conversation_ctx,
            background_tasks,
        ),
        media_type="text/event-stream",
    )


async def stream_agentic_response(
    response_text: str,
    conversation_id: int,
    model_profile: Optional[ModelProfile],
    conversation_ctx: ConversationContext,
    background_tasks: Optional[BackgroundTasks] = None,
) -> AsyncIterable[str]:
    """
    Stream an agentic response that was generated non-streaming

    Args:
        response_text: Full text to stream
        conversation_id: The conversation ID
        model_profile: The model profile
        conversation_ctx: The conversation context
        background_tasks: Optional background tasks for async operations

    Yields:
        Streaming response chunks
    """
    # Split response into chunks for streaming effect
    words = response_text.split()
    chunk_size = 3  # Words per chunk
    model_name = model_profile.model_name if model_profile else "default_model"

    # Stream chunks with small delay to simulate typing
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])

        # Create chunk response
        chunk_response = create_valid_chat_response(
            done=False,
            message=Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(type=MessageContentType.TEXT, text=chunk, url=None)
                ],
            ),
            created_at=dt.now(),
            model=model_name,
        )

        yield f"data: {json.dumps(chunk_response.dict())}\n\n"
        await asyncio.sleep(0.05)  # Small delay

    # Create final message with complete response
    final_message = Message(
        role=MessageRole.ASSISTANT,
        content=[
            MessageContent(type=MessageContentType.TEXT, text=response_text, url=None)
        ],
    )

    # Add thinking if configured in model profile
    if (
        model_profile
        and hasattr(model_profile, "parameters")
        and model_profile.parameters
        and hasattr(model_profile.parameters, "think")
    ):
        final_message.thinking = str(model_profile.parameters.think)

    # Create final response
    final_response = create_valid_chat_response(
        done=True,
        message=final_message,
        created_at=dt.now(),
        model=model_name,
        finish_reason="stop",
    )

    # Send final response
    yield f"data: {json.dumps(final_response.dict())}\n\n"

    # Store the final message in background
    if background_tasks:
        # Create properly formatted message for storage
        storage_message = Message(
            role=MessageRole.ASSISTANT,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT, text=response_text, url=None
                )
            ],
        )
        background_tasks.add_task(
            conversation_ctx.add_assistant_message, storage_message
        )


async def generate_streaming_response(
    pipeline: PipelineProtocol,
    messages: List[Message],
    conversation_id: int,
    model_profile: Optional[ModelProfile],
    conversation_ctx: ConversationContext,
    background_tasks: Optional[BackgroundTasks] = None,
) -> AsyncIterable[str]:
    """
    Generate a streaming response using the model pipeline.

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
    # Get model parameters
    options = _extract_model_parameters(model_profile)
    model_name = model_profile.model_name if model_profile else "default_model"

    full_response = ""
    start_time = dt.now()

    try:
        # Get token stream from model
        async_stream = await pipeline.generate_stream(messages, options)

        # Process each token
        async for token in async_stream:
            full_response += token

            # Create chunk response and send
            yield f"data: {json.dumps(create_valid_chat_response(
                done=False,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[MessageContent(type=MessageContentType.TEXT, text=token, url=None)]
                ),
                created_at=dt.now(),
                model=model_name
            ).dict())}\n\n"

        # Create final message
        final_message = Message(
            role=MessageRole.ASSISTANT,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT, text=full_response, url=None
                )
            ],
        )

        # Add thinking if configured
        if (
            model_profile
            and hasattr(model_profile, "parameters")
            and model_profile.parameters
            and hasattr(model_profile.parameters, "think")
        ):
            final_message.thinking = str(model_profile.parameters.think)

        # Send final response
        final_response = create_valid_chat_response(
            done=True,
            message=final_message,
            created_at=dt.now(),
            model=model_name,
            finish_reason="stop",
            total_duration=(dt.now() - start_time).total_seconds(),
        )
        yield f"data: {json.dumps(final_response.dict())}\n\n"

        # Store message in background
        if background_tasks:
            storage_message = Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT, text=full_response, url=None
                    )
                ],
            )
            background_tasks.add_task(
                conversation_ctx.add_assistant_message, storage_message
            )

    except Exception as e:
        # Handle errors gracefully
        logger.error(f"Error in streaming generation: {e}", exc_info=True)
        error_message = Message(
            role=MessageRole.ASSISTANT,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text=f"Error generating response: {str(e)}",
                    url=None,
                )
            ],
        )
        error_response = create_valid_chat_response(
            done=True,
            message=error_message,
            created_at=dt.now(),
            model="error",
            finish_reason="error",
        )
        yield f"data: {json.dumps(error_response.dict())}\n\n"


def _extract_model_parameters(model_profile: Optional[ModelProfile]) -> Dict[str, Any]:
    """
    Extract parameters from model profile for pipeline use

    Args:
        model_profile: The model profile to extract parameters from

    Returns:
        Dict of parameters
    """
    if not model_profile or not model_profile.parameters:
        return {}

    try:
        # Use Pydantic model_dump() or dict() based on available method
        if hasattr(model_profile.parameters, "model_dump"):
            return model_profile.parameters.model_dump()
        else:
            return model_profile.parameters.dict()
    except (AttributeError, ValueError, TypeError) as e:
        logger.warning(f"Failed to convert parameters to dict: {e}")
        return {}


async def generate_complete_response(
    pipeline: PipelineProtocol,
    messages: List[Message],
    conversation_id: int,
    model_profile: Optional[ModelProfile],
    conversation_ctx: ConversationContext,
) -> ChatResponse:
    """
    Generate a complete non-streaming response.

    NOTE: This function is kept for reference but is no longer used
    since all responses are now streamed.

    Args:
        pipeline: The model pipeline to use
        messages: The enhanced messages
        conversation_id: The conversation ID
        model_profile: The full model profile
        conversation_ctx: The conversation context

    Returns:
        Complete chat response
    """
    # Get model parameters
    options = _extract_model_parameters(model_profile)
    model_name = model_profile.model_name if model_profile else "default_model"

    try:
        # Generate the complete response
        response_content = await pipeline.generate(messages, options)

        # Ensure response content is valid
        if not response_content:
            response_content = "I'm sorry, but I couldn't generate a response."

        # Create assistant message
        message = Message(
            role=MessageRole.ASSISTANT,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT, text=response_content, url=None
                )
            ],
        )

        # Add thinking if configured
        if (
            model_profile
            and hasattr(model_profile, "parameters")
            and model_profile.parameters
            and hasattr(model_profile.parameters, "think")
        ):
            message.thinking = str(model_profile.parameters.think)

        # Store the message
        try:
            await conversation_ctx.add_assistant_message(message)
        except Exception as e:
            logger.error(f"Failed to store assistant message: {e}", exc_info=True)

        # Return the response
        return create_valid_chat_response(
            done=True,
            message=message,
            created_at=dt.now(),
            model=model_name,
            finish_reason="stop",
        )

    except Exception as e:
        logger.error(f"Error generating complete response: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {str(e)}",
        )


# All helper functions now inline in their respective functions
