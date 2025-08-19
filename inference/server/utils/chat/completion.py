"""
Chat completion logic for generating responses, streaming, and handling agentic workflows.
"""

import asyncio
import json
from datetime import datetime as dt
from typing import List, Any, Optional, AsyncIterable, Union

from fastapi import BackgroundTasks, HTTPException, status
from fastapi.responses import StreamingResponse

from inference.server.db import storage
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

# Type definition for pipeline
PipelineProtocol = Any  # Using Any since Protocol definition was causing lint errors


async def enhanced_chat_completion_logic(
    conversation_ctx: ConversationContext,
    background_tasks: Optional[BackgroundTasks] = None,
) -> Union[StreamingResponse, ChatResponse]:
    """
    Enhanced chat completion logic that can use agentic workflow

    This replaces the final response generation in the chat_completion function
    """
    user_message = conversation_ctx.current_user_message
    assert user_message, "Current user message not set"
    # Extract text from user message
    user_text = extract_message_text(user_message).strip()
    # Determine if we should use agentic workflow
    use_agentic = should_use_agentic_workflow(user_text)

    if use_agentic:
        logger.info("Using agentic workflow for response generation")
        # Use agentic workflow
        try:
            mp = await storage.get_service(
                storage.model_profile
            ).get_model_profile_by_id(
                conversation_ctx.user_config.model_profiles.engineering_profile_id,
                conversation_ctx.user_id,
            )
            assert mp, "Model profile not found"

            response_text = await create_agentic_chat_completion(
                conversation_ctx=conversation_ctx,
                model_id=mp.name,
            )

            if stream:
                # For streaming, we need to simulate chunks
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
            else:
                # For non-streaming, create a complete response
                message_data = {
                    "role": MessageRole.ASSISTANT,
                    "content": [
                        MessageContent(
                            type=MessageContentType.TEXT, text=response_text, url=None
                        )
                    ],
                    "conversation_id": conversation_id,
                }

                # Ensure the message is properly formatted
                message = ensure_valid_message(message_data, conversation_id)

                # Store the message
                if background_tasks:

                    async def store_message_task(
                        ctx: ConversationContext, msg: Message
                    ) -> None:
                        try:
                            await ctx.add_assistant_message(msg)
                            logger.info(
                                f"Stored assistant message for conversation {msg.conversation_id}"
                            )
                        except (ValueError, KeyError, AttributeError, IOError) as e:
                            logger.error(
                                f"Failed to store assistant message: {e}", exc_info=True
                            )

                    background_tasks.add_task(
                        store_message_task,
                        conversation_ctx,
                        message,
                    )

                # Create and return the response
                return create_valid_chat_response(
                    done=True,
                    message=message,
                    created_at=dt.now(),
                    model=(
                        model_profile.model_name if model_profile else "default_model"
                    ),
                    finish_reason="stop",
                )
        except Exception as e:
            logger.error(f"Error in agentic workflow: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error in agentic workflow: {str(e)}",
            ) from e

    # Use standard pipeline workflow
    model_id = (
        str(model_profile.id)
        if model_profile and hasattr(model_profile, "id")
        else "default"
    )
    pipeline_result, _ = pipeline_factory_arg.get_pipeline(model_id)
    pipeline: PipelineProtocol = pipeline_result

    if stream:
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
    else:
        return await generate_complete_response(
            pipeline,
            enhanced_messages,
            conversation_id,
            model_profile,
            conversation_ctx,
        )


async def stream_agentic_response(
    response_text: str,
    conversation_id: int,
    model_profile: Optional[ModelProfile],
    conversation_ctx: ConversationContext,
    background_tasks: Optional[BackgroundTasks] = None,
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
        chunk_message = Message(
            role=MessageRole.ASSISTANT,
            content=[
                MessageContent(type=MessageContentType.TEXT, text=chunk, url=None)
            ],
            conversation_id=conversation_id,
        )

        # Create a response for this chunk
        chunk_response = create_valid_chat_response(
            done=False,  # Not done until the final chunk
            message=chunk_message,
            created_at=dt.now(),
            model=model_profile.model_name if model_profile else "default_model",
        )

        # Yield the serialized response
        yield f"data: {json.dumps(chunk_response.dict())}\n\n"

        # Small delay to simulate typing
        await asyncio.sleep(0.05)

    # Send final done message
    final_content = MessageContent(
        type=MessageContentType.TEXT, text=response_text, url=None
    )

    # Create the message and ensure it's valid
    # Convert boolean thinking to string if needed
    thinking_value = None
    if (
        model_profile
        and hasattr(model_profile, "parameters")
        and model_profile.parameters
        and hasattr(model_profile.parameters, "think")
        and model_profile.parameters.think is not None
    ):
        thinking_value = str(model_profile.parameters.think)

    message_data = {
        "role": MessageRole.ASSISTANT,
        "content": [final_content],
        "conversation_id": conversation_id,
        "tool_calls": None,
        "thinking": thinking_value,
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

        async def store_message_task(ctx: ConversationContext, msg: Message) -> None:
            try:
                await ctx.add_assistant_message(msg)
                logger.info(
                    f"Stored assistant message for conversation {msg.conversation_id}"
                )
            except (ValueError, KeyError, AttributeError, IOError) as e:
                logger.error(f"Failed to store assistant message: {e}", exc_info=True)

        background_tasks.add_task(
            store_message_task,
            conversation_ctx,
            final_message,
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
    options = {}
    if model_profile and model_profile.parameters:
        try:
            # Use Pydantic model_dump() or dict() based on available method
            if hasattr(model_profile.parameters, "model_dump"):
                options = model_profile.parameters.model_dump()
            else:
                options = model_profile.parameters.dict()
        except (AttributeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to convert parameters to dict: {e}")
            # Fallback to empty dict if conversion fails

    full_response = ""
    full_message = None
    start_time = dt.now()

    try:
        # Call the async method and await its result before iterating
        async_stream = await pipeline.generate_stream(messages, options)
        async for token in async_stream:
            full_response += token

            # Create a message for this token
            chunk_message = Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(type=MessageContentType.TEXT, text=token, url=None)
                ],
                conversation_id=conversation_id,
            )

            # Create a response for this chunk
            chunk_response = create_valid_chat_response(
                done=False,
                message=chunk_message,
                created_at=dt.now(),
                model=model_profile.model_name if model_profile else "default_model",
            )

            # Yield the serialized response
            yield f"data: {json.dumps(chunk_response.dict())}\n\n"

        # Create the final complete message
        thinking_value = None
        if (
            model_profile
            and model_profile.parameters
            and hasattr(model_profile.parameters, "think")
            and model_profile.parameters.think is not None
        ):
            # Convert boolean thinking to string if needed
            thinking_value = str(model_profile.parameters.think)

        full_message = Message(
            role=MessageRole.ASSISTANT,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT, text=full_response, url=None
                )
            ],
            conversation_id=conversation_id,
            thinking=thinking_value,
        )

        # Create the final response
        final_response = create_valid_chat_response(
            done=True,
            message=full_message,
            created_at=dt.now(),
            model=model_profile.model_name if model_profile else "default_model",
            finish_reason="stop",
            total_duration=(dt.now() - start_time).total_seconds(),
        )

        # Yield the final response
        yield f"data: {json.dumps(final_response.dict())}\n\n"

        # Store the final message directly
        if background_tasks is not None:

            async def store_message_task(
                ctx: ConversationContext, msg: Message
            ) -> None:
                try:
                    await ctx.add_assistant_message(msg)
                    logger.info(
                        f"Stored assistant message for conversation {msg.conversation_id}"
                    )
                except (ValueError, KeyError, AttributeError, IOError) as e:
                    logger.error(
                        f"Failed to store assistant message: {e}", exc_info=True
                    )

            background_tasks.add_task(
                store_message_task,
                conversation_ctx,
                full_message,
            )
    except (ValueError, KeyError, AttributeError, IOError, asyncio.TimeoutError) as e:
        logger.error(f"Error generating streaming response: {e}", exc_info=True)
        error_message = Message(
            role=MessageRole.ASSISTANT,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text=f"Error generating response: {str(e)}",
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
    pipeline: PipelineProtocol,
    messages: List[Message],
    conversation_id: int,
    model_profile: Optional[ModelProfile],
    conversation_ctx: ConversationContext,
) -> ChatResponse:
    """Generate a complete non-streaming response."""
    # Generate the complete response
    # Use model profile to access parameters, system prompt, thinking options, and image settings
    options = {}
    if model_profile and model_profile.parameters:
        try:
            # Use Pydantic model_dump() or dict() based on available method
            if hasattr(model_profile.parameters, "model_dump"):
                options = model_profile.parameters.model_dump()
            else:
                options = model_profile.parameters.dict()
        except (AttributeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to convert parameters to dict: {e}")
            # Fallback to empty dict if conversion fails

    # Pass options to the pipeline
    response_content = await pipeline.generate(messages, options)

    # Create the response message with proper structure
    # Ensure response content is valid
    if not response_content:
        response_content = "I'm sorry, but I couldn't generate a response."

    # Always create a proper MessageContent object
    content_item = MessageContent(
        type=MessageContentType.TEXT, text=response_content, url=None
    )

    # Convert boolean thinking to string if needed
    thinking_value = None
    if (
        model_profile
        and model_profile.parameters
        and hasattr(model_profile.parameters, "think")
        and model_profile.parameters.think is not None
    ):
        thinking_value = str(model_profile.parameters.think)

    # Create the message and ensure it's valid
    message_data = {
        "role": MessageRole.ASSISTANT,
        "content": [content_item],
        "conversation_id": conversation_id,
        "tool_calls": None,
        "thinking": thinking_value,
        "id": None,
        "created_at": dt.now(),
    }

    # Validate the message
    response_message = ensure_valid_message(message_data, conversation_id)

    # Store the message
    try:
        await conversation_ctx.add_assistant_message(response_message)
    except (ValueError, KeyError, AttributeError, IOError) as e:
        logger.error(f"Failed to store assistant message: {e}", exc_info=True)
        # Continue with response creation even if storage fails

    # Create and return a validated ChatResponse
    return create_valid_chat_response(
        done=True,
        message=response_message,
        created_at=dt.now(),
        model=model_profile.model_name if model_profile else "default_model",
        finish_reason="stop",
    )


# Store assistant message function removed - functionality inlined where needed
