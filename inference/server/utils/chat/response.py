"""
Response utility functions for handling ChatResponse creation and validation.
"""

from datetime import datetime as dt
from typing import List, Optional

from models.message import Message
from models.message_role import MessageRole
from models.message_content_type import MessageContentType
from models.message_content import MessageContent
from models.chat_response import ChatResponse
from server.config import logger


def create_valid_chat_response(
    done: bool,
    message: Optional[Message] = None,
    created_at: Optional[dt] = None,
    model: str = "default_model",
    context: Optional[List[List[float]]] = None,
    finish_reason: Optional[str] = None,
    total_duration: Optional[float] = None,
    load_duration: Optional[float] = None,
    prompt_eval_count: Optional[float] = None,
    prompt_eval_duration: Optional[float] = None,
    eval_count: Optional[float] = None,
    eval_duration: Optional[float] = None,
    conversation_id: Optional[int] = None,
) -> ChatResponse:
    """
    Create a valid ChatResponse with strict validation of all fields.
    This is a central function to ensure all ChatResponse objects have the required fields properly set.

    Args:
        done: Whether the generation is complete
        message: The message content and metadata
        created_at: Timestamp when the response was created
        model: The name or identifier of the model used for generation
        context: Vectors representing the tokenized context
        finish_reason: Specific indicator of how or why the generation finished
        total_duration: Total time taken for the entire generation process
        load_duration: Time taken to load the model
        prompt_eval_count: Number of tokens in the prompt that were evaluated
        prompt_eval_duration: Time taken to evaluate the prompt tokens
        eval_count: Total number of tokens evaluated
        eval_duration: Time taken for token evaluation
        conversation_id: Optional explicit conversation ID to use if message doesn't have one

    Returns:
        A valid ChatResponse object with all required fields properly set
    """
    # Ensure we have a created_at timestamp
    if created_at is None:
        created_at = dt.now()

    # Create a valid message if none provided or validate the existing one
    valid_message = None
    if message is None:
        # Create a minimal valid message
        valid_message = Message(
            role=MessageRole.ASSISTANT,
            content=[MessageContent(type=MessageContentType.TEXT, text="", url=None)],
            conversation_id=conversation_id if conversation_id is not None else -1,
        )
    else:
        # Validate and fix the existing message
        try:
            # Ensure content is a list of MessageContent objects
            if not hasattr(message, "content") or message.content is None:
                content_list = [
                    MessageContent(type=MessageContentType.TEXT, text="", url=None)
                ]
            elif not isinstance(message.content, list):
                content_list = [
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=str(message.content),
                        url=None,
                    )
                ]
            else:
                # Ensure each content item is a valid MessageContent
                content_list = []
                for item in message.content:
                    if not isinstance(item, MessageContent):
                        text = str(item) if item is not None else ""
                        content_list.append(
                            MessageContent(
                                type=MessageContentType.TEXT, text=text, url=None
                            )
                        )
                    else:
                        content_list.append(item)

            # Get conversation_id from priority sources:
            # 1. Explicit conversation_id parameter
            # 2. Message's conversation_id attribute
            # 3. Default value (-1)
            conversation_id_value = -1

            # First priority: explicit parameter
            if conversation_id is not None:
                conversation_id_value = conversation_id
            # Second priority: message attribute
            elif (
                hasattr(message, "conversation_id")
                and message.conversation_id is not None
            ):
                conversation_id_value = message.conversation_id

            # Create a new valid message with all required fields
            valid_message = Message(
                role=(
                    message.role
                    if hasattr(message, "role") and message.role
                    else MessageRole.ASSISTANT
                ),
                content=content_list,
                conversation_id=conversation_id_value,
                id=message.id if hasattr(message, "id") and message.id else None,
                created_at=(
                    message.created_at
                    if hasattr(message, "created_at") and message.created_at
                    else dt.now()
                ),
                thinking=message.thinking if hasattr(message, "thinking") else None,
                tool_calls=(
                    message.tool_calls if hasattr(message, "tool_calls") else None
                ),
            )
        except Exception as e:
            logger.error(f"Error validating message: {e}")
            # Create a minimal valid message on error
            valid_message = Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text="Error validating message",
                        url=None,
                    )
                ],
                conversation_id=conversation_id if conversation_id is not None else -1,
            )

    # Create the ChatResponse with all required fields
    try:
        return ChatResponse(
            done=done,
            message=valid_message,
            created_at=created_at,
            model=model if model else "default_model",
            context=context,
            finish_reason=finish_reason,
            total_duration=total_duration,
            load_duration=load_duration,
            prompt_eval_count=prompt_eval_count,
            prompt_eval_duration=prompt_eval_duration,
            eval_count=eval_count,
            eval_duration=eval_duration,
        )
    except Exception as e:
        logger.error(f"Error creating ChatResponse: {e}")
        # Return a minimal valid response
        return ChatResponse(
            done=True,
            message=Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text="Error creating response",
                        url=None,
                    )
                ],
                conversation_id=-1,
            ),
            created_at=dt.now(),
            model="error_model",
            finish_reason="error",
        )
