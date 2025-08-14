"""
Chat router for handling conversations and completions.
This implementation uses LangChain for enhanced RAG capabilities.
Updated to use pipeline factory and intent-based service instantiation.

Note: This router is included in app.py with both non-versioned and versioned paths:
- Non-versioned: /chat/...
- Versioned: /v1/chat/...
"""

import time
import asyncio
import json
import re
from datetime import datetime as dt
from datetime import datetime
from typing import Coroutine, List, Dict, Any, Optional, AsyncIterable, Union

from models.conversation import Conversation
from models.chat_response import ChatResponse
from models.model_profile import ModelProfile
from models.message import Message
from models.message_role import MessageRole
from models.message_content_type import MessageContentType
from models.message_content import MessageContent
from server.config import logger  # Import logger from config
from server.auth import get_user_id, is_admin, get_request_id
from server.db import storage  # Import database storage
from server.context.conversation import (
    ConversationContext,
)  # Import updated ConversationContext

from runner.pipelines.factory import pipeline_factory
from server.tools import create_agentic_chat_completion

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    status,
    BackgroundTasks,
)
from fastapi.responses import StreamingResponse
import json

router = APIRouter(prefix="/chat", tags=["chat"])


def create_valid_chat_response(
    done: bool,
    message: Optional[Message] = None,
    created_at: Optional[datetime] = None,
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

            # Log message validation for debugging
            logger.debug(
                f"Validating message with conversation_id: {conversation_id_value}"
            )

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


def should_use_agentic_workflow(user_message: str) -> bool:
    """
    Determine if a user message would benefit from agentic processing with tools

    Args:
        user_message: The user's message text

    Returns:
        bool: True if agentic workflow should be used
    """
    # Keywords that suggest need for tools/computation
    tool_indicators = [
        # Calculation keywords
        "calculate",
        "compute",
        "add",
        "subtract",
        "multiply",
        "divide",
        "sum",
        "average",
        "mean",
        "median",
        "standard deviation",
        "percentage",
        "percent",
        "ratio",
        "proportion",
        # Data processing keywords
        "analyze",
        "process",
        "transform",
        "convert",
        "parse",
        "filter",
        "sort",
        "group",
        "aggregate",
        "summarize",
        # Programming/algorithm keywords
        "algorithm",
        "function",
        "code",
        "script",
        "program",
        "logic",
        "formula",
        "equation",
        "solve",
        # Complex task indicators
        "step by step",
        "break down",
        "systematic",
        "methodical",
        "optimize",
        "find the best",
        "compare options",
    ]

    # Check for mathematical expressions
    math_patterns = [
        r"\d+\s*[+\-*/]\s*\d+",  # Basic math operations
        r"\d+\s*%",  # Percentages
        r"\$\d+",  # Currency
        r"\d+\.\d+",  # Decimals
    ]

    message_lower = user_message.lower()

    # Check for tool indicator keywords
    for indicator in tool_indicators:
        if indicator in message_lower:
            return True

    # Check for mathematical patterns
    for pattern in math_patterns:
        if re.search(pattern, user_message):
            return True

    # Check for question words that might need computation
    computation_questions = [
        "how many",
        "how much",
        "what is the",
        "calculate the",
        "find the",
        "determine the",
        "compute the",
    ]

    for question in computation_questions:
        if question in message_lower:
            return True

    return False


def extract_message_text(message: Message) -> str:
    """Extract text content from a message object"""
    text_parts = []
    for content in message.content:
        if content.type == MessageContentType.TEXT and content.text:
            text_parts.append(content.text)
    return " ".join(text_parts).strip()


def convert_string_to_message_content(message_text: str) -> List[MessageContent]:
    """Convert a plain string to a list of MessageContent objects"""
    return [MessageContent(type=MessageContentType.TEXT, text=message_text, url=None)]


def ensure_valid_message(message, default_conversation_id: int = -1):
    """
    Ensure a message object is valid by checking and fixing both content and conversation_id.

    Args:
        message: The message object to validate
        default_conversation_id: The default conversation ID to use if none is provided

    Returns:
        A valid message object with proper content format and conversation_id
    """
    # If message is a dict, convert to Message object
    if isinstance(message, dict):
        # Extract values from dict
        role = message.get("role", MessageRole.USER)
        content = message.get("content", "")

        # Ensure conversation_id is set - this is critical
        conversation_id = message.get("conversation_id")
        if conversation_id is None:
            conversation_id = message.get("id_conversation")  # Try alternate format
        if conversation_id is None:
            conversation_id = default_conversation_id  # Fall back to default

        message_id = message.get("message_id", message.get("id"))
        created_at = message.get("created_at", message.get("timestamp", dt.now()))
        thinking = message.get("thinking")
        tool_calls = message.get("tool_calls")

        # Format content as MessageContent list
        if isinstance(content, str):
            content = convert_string_to_message_content(content)
        elif isinstance(content, list) and all(
            isinstance(item, MessageContent) for item in content
        ):
            # Content is already in correct format
            pass
        elif isinstance(content, list):
            # List of something else - convert each item
            formatted_content = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    formatted_content.append(
                        MessageContent(
                            type=item.get("type", MessageContentType.TEXT),
                            text=item["text"],
                            url=item.get("url"),
                        )
                    )
                else:
                    text = str(item) if item is not None else ""
                    formatted_content.append(
                        MessageContent(
                            type=MessageContentType.TEXT, text=text, url=None
                        )
                    )
            content = formatted_content
        else:
            # Some other format - convert to string
            try:
                text = str(content) if content is not None else ""
                content = convert_string_to_message_content(text)
            except:
                content = [
                    MessageContent(type=MessageContentType.TEXT, text="", url=None)
                ]

        # Ensure we have a valid conversation_id before creating the Message
        # This is absolutely critical since conversation_id is a required field
        if conversation_id is None or not isinstance(conversation_id, int):
            logger.warning(
                f"Invalid conversation_id found: {conversation_id}, using default: {default_conversation_id}"
            )
            conversation_id = default_conversation_id

        # Create new Message object with required fields
        try:
            message = Message(
                role=role,
                content=content,
                conversation_id=conversation_id,
                id=message_id,
                created_at=created_at,
                thinking=thinking,
                tool_calls=tool_calls,
            )
        except Exception as e:
            # If creation fails, try with minimal required fields
            logger.warning(f"Failed to create message with all fields: {e}")
            message = Message(
                role=MessageRole.USER if not isinstance(role, MessageRole) else role,
                content=[
                    MessageContent(type=MessageContentType.TEXT, text="", url=None)
                ],
                conversation_id=default_conversation_id,
            )

    # Now message should be a Message object
    # Ensure content is a list of MessageContent objects
    try:
        if message.content is None:
            # Handle None content
            logger.warning("Message content is None, setting to empty MessageContent")
            message.content = [
                MessageContent(type=MessageContentType.TEXT, text="", url=None)
            ]
        elif not isinstance(message.content, list):
            # If content is a string, convert it to a list with one MessageContent
            if isinstance(message.content, str):
                logger.info(f"Converting string content to MessageContent list")
                message.content = convert_string_to_message_content(message.content)
            else:
                # Handle other types
                try:
                    logger.info(
                        f"Converting non-string non-list content to MessageContent list: {type(message.content)}"
                    )
                    text = str(message.content) if message.content is not None else ""
                    message.content = convert_string_to_message_content(text)
                except Exception as e:
                    logger.warning(
                        f"Error converting content to MessageContent list: {e}"
                    )
                    message.content = [
                        MessageContent(type=MessageContentType.TEXT, text="", url=None)
                    ]
    except Exception as e:
        # Fallback for any unexpected errors
        logger.error(f"Unexpected error handling message content: {e}")
        message.content = [
            MessageContent(type=MessageContentType.TEXT, text="", url=None)
        ]

    # Ensure each item in the list is a MessageContent
    for i, content in enumerate(message.content):
        if not isinstance(content, MessageContent):
            try:
                # Try to convert to MessageContent
                text = (
                    content.get("text", "")
                    if isinstance(content, dict)
                    else str(content)
                )
                message.content[i] = MessageContent(
                    type=MessageContentType.TEXT, text=text, url=None
                )
            except:
                message.content[i] = MessageContent(
                    type=MessageContentType.TEXT, text="", url=None
                )

    # Ensure conversation_id exists and is valid - be very aggressive about setting this
    if not hasattr(message, "conversation_id"):
        message.conversation_id = default_conversation_id
        logger.warning(
            f"Message missing conversation_id attribute, setting to default: {default_conversation_id}"
        )
    elif message.conversation_id is None:
        message.conversation_id = default_conversation_id
        logger.warning(
            f"Message has None conversation_id, setting to default: {default_conversation_id}"
        )
    elif not isinstance(message.conversation_id, int):
        try:
            # Try to convert to int
            message.conversation_id = int(message.conversation_id)
            logger.warning(
                f"Converted non-int conversation_id to int: {message.conversation_id}"
            )
        except:
            message.conversation_id = default_conversation_id
            logger.warning(
                f"Failed to convert conversation_id to int, using default: {default_conversation_id}"
            )

    # Final check - if for any reason conversation_id is still not set, use default
    try:
        if (
            not message.conversation_id and message.conversation_id != 0
        ):  # This would catch None or empty values
            message.conversation_id = (
                default_conversation_id if default_conversation_id != 0 else -1
            )
            logger.warning(
                f"Final check: Empty conversation_id, set to: {message.conversation_id}"
            )
    except Exception as e:
        message.conversation_id = (
            default_conversation_id if default_conversation_id != 0 else -1
        )
        logger.warning(
            f"Final check exception: {e}, set conversation_id to: {message.conversation_id}"
        )

    return message


def ensure_message_content_list(message):
    """Ensure message content is a list of MessageContent objects"""
    if not isinstance(message.content, list):
        # If content is a string, convert it to a list with one MessageContent
        if isinstance(message.content, str):
            message.content = convert_string_to_message_content(message.content)
        else:
            # Handle other types (like dict)
            try:
                text = str(message.content)
                message.content = convert_string_to_message_content(text)
            except:
                message.content = [
                    MessageContent(type=MessageContentType.TEXT, text="", url=None)
                ]

    # Ensure each item in the list is a MessageContent
    for i, content in enumerate(message.content):
        if not isinstance(content, MessageContent):
            try:
                # Try to convert to MessageContent
                text = (
                    content.get("text", "")
                    if isinstance(content, dict)
                    else str(content)
                )
                message.content[i] = MessageContent(
                    type=MessageContentType.TEXT, text=text, url=None
                )
            except:
                message.content[i] = MessageContent(
                    type=MessageContentType.TEXT, text="", url=None
                )

    return message


@router.post("/completions", response_model=ChatResponse)
async def chat_completion(
    request: Request,
    background_tasks: BackgroundTasks,
    stream: bool = False,
):
    """
    Handle chat completions by processing a single user message and generating a response.
    This implementation uses LangChain for enhanced RAG capabilities including:

    1. Document retrieval from PostgreSQL vector store
    2. Web search integration (only when intent.web_search is True)
    3. URL content extraction
    4. Reranking of retrieved documents
    5. Deduplication of information
    6. Context-aware summarization
    7. Enhanced prompt creation with retrieved contexts
    8. Streaming or complete response generation

    The function integrates with the Pipeline Factory for all model access.
    """
    # Set up request context information
    user_id = get_user_id(request)
    request_id = get_request_id(request)

    # Get raw request body first to properly handle all formats
    try:
        raw_body = await request.json()
        logger.debug(f"Raw request body: {raw_body}")

        # Extract required fields from raw body
        raw_role = raw_body.get("role", "user")
        raw_content = raw_body.get("content", "")

        # Extract conversation_id with fallbacks
        # First try from the request body
        raw_conversation_id = raw_body.get("conversation_id")

        # If not found in the body, try to extract from the URL path
        if raw_conversation_id is None:
            try:
                # Extract conversation_id from URL path like /chat/conversations/123/messages
                path = request.url.path
                import re

                match = re.search(r"/conversations/(\d+)", path)
                if match:
                    raw_conversation_id = int(match.group(1))
                    logger.info(
                        f"Extracted conversation_id {raw_conversation_id} from URL path: {path}"
                    )
            except Exception as e:
                logger.warning(f"Failed to extract conversation_id from URL: {e}")

        raw_message_id = raw_body.get("message_id", raw_body.get("id"))
        raw_timestamp = raw_body.get("timestamp", raw_body.get("created_at", dt.now()))

        # Ensure content is a list of MessageContent objects
        content_list = []
        if isinstance(raw_content, list):
            for item in raw_content:
                if isinstance(item, dict) and "text" in item:
                    content_list.append(
                        MessageContent(
                            type=item.get("type", MessageContentType.TEXT),
                            text=item["text"],
                            url=item.get("url"),
                        )
                    )
                else:
                    content_list.append(
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=str(item) if item is not None else "",
                            url=None,
                        )
                    )
        else:
            # Convert string or other type to MessageContent
            content_list = [
                MessageContent(
                    type=MessageContentType.TEXT,
                    text=str(raw_content) if raw_content is not None else "",
                    url=None,
                )
            ]

        # Set a default conversation_id if none provided
        if raw_conversation_id is None:
            raw_conversation_id = -1
            logger.warning(
                f"No conversation_id provided in request, using default: {raw_conversation_id}"
            )

        # Create a valid Message object
        message = Message(
            role=raw_role,
            content=content_list,
            conversation_id=raw_conversation_id,
            id=raw_message_id,
            created_at=raw_timestamp,
        )

        logger.info(f"Created validated message from raw request: {message}")
    except Exception as e:
        logger.error(f"Error processing raw request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request format: {str(e)}",
        )

    # Log the start of request processing
    logger.info(f"Processing chat completion request {request_id} for user {user_id}")

    # Double-check and validate message content
    if not message.role:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message role cannot be empty",
        )
    if not message.content:
        logger.warning("Empty message content, creating default empty content")
        message.content = [
            MessageContent(type=MessageContentType.TEXT, text="", url=None)
        ]

    # Ensure content is actually a list
    if not isinstance(message.content, list):
        logger.warning(
            f"Message content is not a list: {type(message.content)}, converting"
        )
        if isinstance(message.content, str):
            message.content = [
                MessageContent(
                    type=MessageContentType.TEXT, text=message.content, url=None
                )
            ]
        else:
            try:
                message.content = [
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=str(message.content),
                        url=None,
                    )
                ]
            except:
                message.content = [
                    MessageContent(type=MessageContentType.TEXT, text="", url=None)
                ]

    # Validate each content item
    for i, content in enumerate(message.content):
        if not isinstance(content, MessageContent):
            logger.warning(
                f"Content item at index {i} is not MessageContent, converting"
            )
            try:
                text = (
                    content.get("text", "")
                    if isinstance(content, dict)
                    else str(content)
                )
                message.content[i] = MessageContent(
                    type=MessageContentType.TEXT, text=text, url=None
                )
            except:
                message.content[i] = MessageContent(
                    type=MessageContentType.TEXT, text="", url=None
                )
        elif not content.type:
            logger.warning(f"Content item at index {i} has no type, setting default")
            content.type = MessageContentType.TEXT

    # Final check for conversation_id
    if not hasattr(message, "conversation_id") or message.conversation_id is None:
        logger.warning("Message missing conversation_id, setting default -1")
        message.conversation_id = -1

    # Verify message is from user
    if message.role != MessageRole.USER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Message must be from user"
        )

    # Get conversation ID from message
    conversation_id = message.conversation_id
    conversation_ctx = None

    # Ensure user is authenticated
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )

    if (
        not storage.user_config
        or not storage.conversation
        or not storage.model_profile
        or not storage.message
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )

    # Fetch user's model profile configuration
    user_config = await storage.user_config.get_user_config(user_id)
    if not user_config or not user_config.model_profiles:
        logger.warning(f"No model profile configuration found for user {user_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User model profile configuration not found",
        )

    # Get model profiles from user config
    model_profiles = user_config.model_profiles

    # Get profile IDs from user config
    embedding_profile_id = str(model_profiles.embedding_profile_id)
    summarization_profile_id = str(model_profiles.summarization_profile_id)

    logger.info(f"Using embedding profile: {embedding_profile_id}")
    logger.info(f"Using summarization profile: {summarization_profile_id}")

    if conversation_id == -1:
        # Special case for no conversation context (e.g. image generation)
        logger.info(
            "Using conversation_id=-1, creating temporary context without history"
        )

        # Create a temporary conversation context with no history
        conversation_ctx = ConversationContext(
            user_id=user_id,
            conversation_id=-1,  # Use -1 as the temporary ID
            embedding_profile_id=embedding_profile_id,
            summarization_profile_id=summarization_profile_id,
            user_config=user_config,
        )

    elif conversation_id:
        # Load existing conversation context
        try:

            # Verify conversation ownership
            conversation = await storage.conversation.get_conversation(conversation_id)
            if not conversation or conversation.user_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not authorized to access this conversation",
                )

            # Create context with pipeline factory IDs
            conversation_ctx = ConversationContext(
                user_id=user_id,
                conversation_id=conversation_id,
                embedding_profile_id=embedding_profile_id,
                summarization_profile_id=summarization_profile_id,
                user_config=user_config,
            )
            await conversation_ctx.load_conversation_data()
            logger.info(
                f"Loaded existing conversation context for conversation {conversation_id}"
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to load conversation context: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load conversation context: {str(e)}",
            ) from e
    else:
        # Create a new conversation
        try:
            # Create the conversation with a temporary title
            new_conversation_id = await storage.conversation.create_conversation(
                user_id, "New conversation"
            )
            if not new_conversation_id:
                raise ValueError("Failed to create conversation: no ID returned")

            conversation_id = new_conversation_id

            # Create context with pipeline factory IDs
            conversation_ctx = ConversationContext(
                user_id=user_id,
                conversation_id=conversation_id,
                embedding_profile_id=embedding_profile_id,
                summarization_profile_id=summarization_profile_id,
                user_config=user_config,
            )

            # Get formatting profile ID from model profiles
            formatting_profile_id = str(model_profiles.formatting_profile_id)
            logger.info(f"Using formatting profile: {formatting_profile_id}")

            # Generate title using formatting model
            generated_title = await conversation_ctx.generate_title_with_formatting(
                message, formatting_profile_id
            )

            # Update the title with the generated one
            await storage.conversation.update_conversation_title(
                conversation_id, generated_title
            )

            logger.info(
                f"Created new conversation with ID {conversation_id} and title '{generated_title}'"
            )
            logger.info(f"Created new conversation with ID {conversation_id}")

            # Update the message's conversation_id to match the new conversation
            message.conversation_id = conversation_id
        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create conversation: {str(e)}",
            ) from e

    # Process the user message directly - no need to extract from a list
    user_message = message

    # Process user message with enhanced RAG
    try:
        embeddings, message_id = await conversation_ctx.add_user_message(user_message)
        if not embeddings:
            logger.warning(f"Empty embedding vector for message {message_id}")

        summarization_task = conversation_ctx.summarize_messages()
        query = next(
            (
                c.text
                for c in user_message.content
                if c.type == MessageContentType.TEXT and c.text
            ),
            "",
        )
        memory_task = (
            conversation_ctx.retrieve_memories(query)
            if conversation_ctx.intent.memory and query
            else None
        )
        web_task = (
            conversation_ctx.search_web(query)
            if conversation_ctx.intent.web_search and query
            else None
        )
        # Create a list for asyncio.gather
        tasks: list[Coroutine[Any, Any, Any]] = [summarization_task]
        if memory_task:
            tasks.append(memory_task)
        if web_task:
            tasks.append(web_task)
        rag_results = await asyncio.gather(*tasks)
        summarization_result = rag_results[0] if rag_results else None
        rag_data = {}
        idx = 1
        if memory_task:
            rag_data["memories"] = rag_results[idx]
            idx += 1
        if web_task:
            rag_data["web_results"] = rag_results[idx] if len(rag_results) > idx else []
        logger.info("RAG preparation completed")

        # Get primary pipeline from factory using the user's model profile
        # Use the primary model profile from user config
        model_id = str(model_profiles.primary_profile_id)
        logger.info(f"Using primary model profile: {model_id}")

        # Get the appropriate pipeline from the factory
        pipeline, load_time = pipeline_factory.get_pipeline(model_id)

        if not pipeline:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model {model_id} not available",
            )

        logger.info(f"Model pipeline loaded in {load_time:.2f}ms")

        # Prepare the final prompt with context
        # For conversation_id = -1, we won't have any previous messages
        # Otherwise, we'll need to get previous messages from conversation context
        previous_messages = []
        if conversation_id != -1:
            # Get previous messages from the conversation
            conversation_messages = await storage.message.get_conversation_history(
                conversation_id
            )
            previous_messages = conversation_messages

        # Add the current user message at the end
        all_messages = previous_messages + [user_message]

        # Get the full model profile to access system prompt, thinking mode, and parameters
        model_profile = None
        try:
            model_profile = await storage.model_profile.get_model_profile_by_id(
                model_profiles.primary_profile_id, user_id
            )
            if not model_profile:
                logger.warning(
                    f"Model profile {model_profiles.primary_profile_id} not found"
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model profile {model_profiles.primary_profile_id} not found",
                )
        except Exception as e:
            logger.warning(f"Failed to get model profile: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to get model profile: {str(e)}",
            ) from e

        # Prepare the final prompt with context
        enhanced_messages = prepare_enhanced_messages(
            all_messages,
            summarization_result,
            rag_data,
            conversation_ctx,
            model_profile,  # Pass the model profile to include system prompt
        )

        # Use enhanced chat completion logic which determines whether to use agentic workflow
        return await enhanced_chat_completion_logic(
            user_message=user_message,
            conversation_ctx=conversation_ctx,
            pipeline_factory_arg=pipeline_factory,
            model_profile=model_profile,
            enhanced_messages=enhanced_messages,
            conversation_id=conversation_id,
            user_id=user_id,
            stream=stream,
            background_tasks=background_tasks,
        )

    except Exception as e:  # noqa: BLE001
        logger.error(f"Error in chat completion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in chat completion: {str(e)}",
        ) from e


@router.get("/admin")
async def admin_only(request: Request):
    """
    Admin-only endpoint to demonstrate role-based access control.
    Only users with admin privileges can access this endpoint.
    """
    # Check if user is admin
    if not is_admin(request):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required for this endpoint",
        )

    user_id = get_user_id(request)
    request_id = get_request_id(request)

    logger.info(f"Admin access granted for user {user_id}, request {request_id}")

    return {
        "status": "success",
        "message": "Admin access granted",
        "user_id": user_id,
        "request_id": request_id,
    }


# Helper functions for RAG, LLM interactions, and agentic workflow


async def enhanced_chat_completion_logic(
    user_message: Message,
    conversation_ctx: ConversationContext,
    pipeline_factory_arg,  # Renamed to avoid shadowing imported pipeline_factory
    model_profile: Optional[ModelProfile],
    enhanced_messages: List[Message],
    conversation_id: int,
    user_id: str,  # noqa: ARG001 - Required to match signature
    stream: bool = False,
    background_tasks: Optional[BackgroundTasks] = None,
) -> Union[StreamingResponse, ChatResponse]:
    """
    Enhanced chat completion logic that can use agentic workflow

    This replaces the final response generation in the chat_completion function
    """
    # Extract text from user message
    user_text = extract_message_text(user_message)

    # Determine if we should use agentic workflow
    use_agentic = should_use_agentic_workflow(user_text)

    if use_agentic:
        logger.info("Using agentic workflow for complex request")

        try:
            # Use agentic workflow
            model_id = (
                str(model_profile.id)
                if model_profile and hasattr(model_profile, "id")
                else "default"
            )

            if stream:
                # For streaming with agentic workflow, we run agentic workflow then stream the result
                agentic_response = await create_agentic_chat_completion(
                    pipeline_factory_arg, conversation_ctx, user_text, model_id
                )

                # Convert to streaming format
                return StreamingResponse(
                    stream_agentic_response(
                        agentic_response,
                        conversation_id,
                        model_profile,
                        conversation_ctx,
                        background_tasks,
                    ),
                    media_type="text/event-stream",
                )
            else:
                # Non-streaming agentic response
                agentic_response = await create_agentic_chat_completion(
                    pipeline_factory_arg, conversation_ctx, user_text, model_id
                )

                # Create response message
                content_item = MessageContent(
                    type=MessageContentType.TEXT, text=agentic_response, url=None
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
                response_message = ensure_valid_message(message_data, conversation_id)

                complete_response = create_valid_chat_response(
                    done=True,
                    message=response_message,
                    created_at=dt.now(),
                    model=(
                        model_profile.model_name if model_profile else "default_model"
                    ),
                    finish_reason="stop",
                )

                # Store in background
                if complete_response.message and background_tasks is not None:
                    background_tasks.add_task(
                        store_assistant_message,
                        conversation_ctx,
                        complete_response.message,
                    )

                return complete_response

        except Exception as e:  # noqa: BLE001
            logger.error(f"Agentic workflow failed, falling back to standard: {e}")
            # Fall back to standard workflow
            use_agentic = False

            # If we're here, either use_agentic was False or we had an exception and fell back
    # Use standard pipeline workflow
    pipeline, _ = pipeline_factory_arg.get_pipeline(
        str(model_profile.id)
        if model_profile and hasattr(model_profile, "id")
        else "default"
    )

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
        complete_response = await generate_complete_response(
            pipeline,
            enhanced_messages,
            conversation_id,
            model_profile,
            conversation_ctx,
        )

        # Store in background
        if complete_response.message and background_tasks is not None:
            background_tasks.add_task(
                store_assistant_message, conversation_ctx, complete_response.message
            )

        return complete_response


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
        chunk_words = words[i : i + chunk_size]
        chunk_text = " " + " ".join(chunk_words)

        content_item = MessageContent(
            type=MessageContentType.TEXT, text=chunk_text, url=None
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
            "id": message_id,
            "created_at": dt.now(),
        }
        chunk_message = ensure_valid_message(message_data, conversation_id)

        response = create_valid_chat_response(
            done=False,
            message=chunk_message,
            created_at=dt.now(),
            model=model_profile.model_name if model_profile else "default_model",
            finish_reason=None,
            conversation_id=conversation_id,  # Explicitly pass conversation_id
        )

        yield f"data: {json.dumps(response.dict())}\n\n"

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
        "id": None,
        "created_at": dt.now(),
    }
    # Ensure final message has valid content and conversation_id
    final_message = ensure_valid_message(message_data, conversation_id)

    # Double-check that content is a list and conversation_id is set
    if not isinstance(final_message.content, list):
        logger.warning(
            f"Final message content is not a list: {type(final_message.content)}"
        )
        final_message.content = [
            MessageContent(
                type=MessageContentType.TEXT, text=str(final_message.content), url=None
            )
        ]

    if (
        not hasattr(final_message, "conversation_id")
        or final_message.conversation_id is None
    ):
        logger.warning("Final message missing conversation_id")
        final_message.conversation_id = (
            conversation_id if conversation_id is not None else -1
        )

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


def format_search_query(query: str) -> str:
    """
    Format the user query for better search results.

    Args:
        query: The raw user query

    Returns:
        A formatted query for web search
    """
    # Remove any specific instructions to the AI
    query = re.sub(r"(?i)please\s+", "", query)
    query = re.sub(r"(?i)can you\s+", "", query)
    query = re.sub(r"(?i)I want you to\s+", "", query)
    query = re.sub(r"(?i)I'd like you to\s+", "", query)

    # Remove unnecessary punctuation
    query = re.sub(r"[^\w\s\?\.]", " ", query)

    # Collapse multiple spaces
    query = re.sub(r"\s+", " ", query).strip()

    # Limit length
    if len(query) > 100:
        query = query[:100]

    return query


def parse_ddg_results(ddg_results: str) -> List[Dict[str, str]]:
    """
    Parse DuckDuckGo search results string into structured data.

    Args:
        ddg_results: String output from DuckDuckGo search

    Returns:
        List of dictionaries with title, link, and snippet
    """
    results = []

    # Simple parsing for the form that DDG usually returns
    try:
        # Handle the case where DDG returns a list of snippets
        lines = ddg_results.split("\n")
        current_item = {}

        for line in lines:
            line = line.strip()
            if not line:
                if current_item and "title" in current_item:
                    results.append(current_item)
                    current_item = {}
                continue

            # Check if this line looks like a URL
            if line.startswith("http"):
                current_item["link"] = line
            # Check if this might be a title (shorter line)
            elif len(line) < 100 and not current_item.get("title"):
                current_item["title"] = line
            # Otherwise assume it's snippet content
            else:
                if "snippet" in current_item:
                    current_item["snippet"] += " " + line
                else:
                    current_item["snippet"] = line

        # Add the last item if it exists
        if current_item and "title" in current_item:
            results.append(current_item)
    except Exception as e:
        logger.warning(f"Error parsing DDG results: {e}")
        # Fallback for unparseable results
        results.append(
            {
                "title": "Search Results",
                "link": "",
                "snippet": ddg_results[:1000],  # Limit length to prevent issues
            }
        )

    return results


def prepare_enhanced_messages(
    messages: List[Message],
    summary: Optional[Any],
    rag_data: Dict[str, Any],
    conversation_ctx: ConversationContext,  # noqa: ARG001, unused but required for signature
    model_profile: Optional[ModelProfile],
) -> List[Message]:
    """
    Prepare enhanced messages with context from RAG results and conversation summaries.

    Args:
        messages: Original message list
        summary: Conversation summary if available
        rag_data: RAG results (memories, web_results, etc.)
        conversation_ctx: Conversation context
        model_profile: Full model profile

    Returns:
        Enhanced message list with system messages for context
    """
    enhanced_messages = []

    # If we have a model profile with system prompt, use it
    if model_profile and model_profile.system_prompt:
        # Create a system message with the system prompt
        system_content = MessageContent(
            type=MessageContentType.TEXT, text=model_profile.system_prompt, url=None
        )
        # Create system message and ensure it's valid
        message_data = {
            "role": MessageRole.SYSTEM,
            "content": [system_content],
            "conversation_id": -1,
            "tool_calls": None,
            "thinking": None,
            "id": None,
            "created_at": dt.now(),
        }
        system_message = ensure_valid_message(message_data, -1)
        enhanced_messages.append(system_message)

    # Add RAG context as a system message if we have any
    rag_contexts = []

    # Add conversation summary if available
    if summary:
        rag_contexts.append(f"CONVERSATION SUMMARY:\n{summary.content}")

    # Add memories if available
    if "memories" in rag_data and rag_data["memories"]:
        memory_texts = []
        for memory in rag_data["memories"]:
            if hasattr(memory, "text") and memory.text:
                memory_texts.append(memory.text)
            elif hasattr(memory, "content") and memory.content:
                memory_texts.append(memory.content)
            elif hasattr(memory, "page_content") and memory.page_content:
                memory_texts.append(memory.page_content)

        if memory_texts:
            rag_contexts.append("RELEVANT MEMORIES:\n" + "\n\n".join(memory_texts))

    # Add web search results if available
    if "web_results" in rag_data and rag_data["web_results"]:
        web_texts = []
        for result in rag_data["web_results"]:
            if hasattr(result, "page_content") and result.page_content:
                # For Document objects
                content = result.page_content
                source = ""
                if hasattr(result, "metadata") and result.metadata:
                    if "source" in result.metadata:
                        source = f" (Source: {result.metadata['source']})"
                    elif "title" in result.metadata:
                        source = f" (Source: {result.metadata['title']})"
                web_texts.append(f"{content}{source}")
            elif hasattr(result, "text") and result.text:
                # For custom result objects
                web_texts.append(result.text)

        if web_texts:
            rag_contexts.append("WEB SEARCH RESULTS:\n" + "\n\n".join(web_texts))

    # Add URL content if available
    if "url_content" in rag_data and rag_data["url_content"]:
        url_texts = []
        for content in rag_data["url_content"]:
            if hasattr(content, "page_content") and content.page_content:
                # For Document objects
                url_text = content.page_content
                source = ""
                if hasattr(content, "metadata") and content.metadata:
                    if "source" in content.metadata:
                        source = f" (Source: {content.metadata['source']})"
                url_texts.append(f"{url_text}{source}")
            elif hasattr(content, "text") and content.text:
                # For custom content objects
                url_texts.append(content.text)

        if url_texts:
            rag_contexts.append("URL CONTENT:\n" + "\n\n".join(url_texts))

    # If we have any RAG contexts, add them as a system message
    if rag_contexts:
        rag_text = "\n\n".join(rag_contexts)
        rag_content = MessageContent(
            type=MessageContentType.TEXT, text=rag_text, url=None
        )
        # Create the RAG message and ensure it's valid
        message_data = {
            "role": MessageRole.SYSTEM,
            "content": [rag_content],
            "conversation_id": -1,
            "tool_calls": None,
            "thinking": None,
            "id": None,
            "created_at": dt.now(),
        }
        rag_message = ensure_valid_message(message_data, -1)
        enhanced_messages.append(rag_message)

    # Add all the original messages
    enhanced_messages.extend(messages)

    return enhanced_messages


async def generate_streaming_response(
    pipeline,
    messages: List[Message],
    conversation_id: int,
    model_profile: Optional[ModelProfile],
    conversation_ctx: ConversationContext,  # noqa: ARG001, unused but required for signature
    background_tasks: Optional[BackgroundTasks] = None,
) -> AsyncIterable[str]:  # noqa: ARG001 for unused user_id
    """
    Generate a streaming response.

    Args:
        pipeline: The model pipeline to use
        messages: The enhanced messages
        conversation_id: The conversation ID
        user_id: The user ID
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

    from models.message_content import MessageContent

    full_response = ""
    # ...existing code...

    try:
        # Stream the response from the pipeline
        async for chunk in pipeline.generate_stream(messages, options):
            full_response += chunk

            # Create a chunk response
            content_item = MessageContent(
                type=MessageContentType.TEXT, text=chunk, url=None
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
            chunk_message = ensure_valid_message(message_data, conversation_id)

            # Convert to ChatResponse
            response = create_valid_chat_response(
                done=False,
                message=chunk_message,
                created_at=dt.now(),
                model=model_profile.model_name if model_profile else "default_model",
                finish_reason=None,
            )

            # Encode as server-sent event
            yield f"data: {json.dumps(response.dict())}\n\n"

        # Send final "done" event
        final_content = MessageContent(
            type=MessageContentType.TEXT, text=full_response, url=None
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
            "id": None,
            "created_at": dt.now(),
        }
        final_message = ensure_valid_message(message_data, conversation_id)

        final_response = create_valid_chat_response(
            done=True,
            message=final_message,
            created_at=dt.now(),
            model=model_profile.model_name if model_profile else "default_model",
            finish_reason="stop",
        )

        yield f"data: {json.dumps(final_response.dict())}\n\n"

        # Store the final message in the background
        if (
            final_message
            and final_message.content
            and len(final_message.content) > 0
            and background_tasks is not None
        ):
            background_tasks.add_task(
                store_assistant_message, conversation_ctx, final_message
            )

    except Exception as e:  # noqa: BLE001
        logger.error(f"Error in streaming response: {e}", exc_info=True)
        error_response = create_valid_chat_response(
            done=True,
            message=None,  # Will create a valid default message
            created_at=dt.now(),
            model=model_profile.model_name if model_profile else "default_model",
            finish_reason="error",
        )
        yield f"data: {json.dumps(error_response.dict())}\n\n"


async def generate_complete_response(
    pipeline,
    messages: List[Message],
    conversation_id: int,
    model_profile: Optional[ModelProfile],
    conversation_ctx: ConversationContext,  # noqa: ARG001, unused but required for signature
) -> ChatResponse:  # noqa: ARG001 for unused user_id
    """Generate a complete non-streaming response."""
    # Generate the complete response
    # Use model profile to access parameters, system prompt, thinking options, and image settings
    options = (
        model_profile.parameters if model_profile and model_profile.parameters else {}
    )

    # Pass options to the pipeline
    response_content = await pipeline.generate(messages, options)

    # Create the response message with proper structure

    # Ensure response content is valid
    if not response_content:
        response_content = ""

    # Always create a proper MessageContent object
    content_item = MessageContent(
        type=MessageContentType.TEXT, text=response_content, url=None
    )

    # Create the message and ensure it's valid
    message_data = {
        "role": MessageRole.ASSISTANT,
        "content": [content_item],  # Always use a list of MessageContent objects
        "conversation_id": conversation_id,  # Ensure conversation_id is set
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

    # Double check that response_message.content is a list
    if not isinstance(response_message.content, list):
        logger.warning(
            f"Response message content is not a list after validation: {type(response_message.content)}"
        )
        response_message.content = [
            MessageContent(
                type=MessageContentType.TEXT,
                text=str(response_message.content),
                url=None,
            )
        ]

    # Double check that conversation_id is set
    if (
        not hasattr(response_message, "conversation_id")
        or response_message.conversation_id is None
    ):
        logger.warning(f"Response message missing conversation_id after validation")
        response_message.conversation_id = conversation_id

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
        # Get the conversation ID from context
        default_conversation_id = -1
        if conversation_ctx and hasattr(conversation_ctx, "conversation_id"):
            default_conversation_id = conversation_ctx.conversation_id

        # Ensure the message is valid and has conversation_id set
        message = ensure_valid_message(message, default_conversation_id)

        # Double-check that conversation_id is set
        if not hasattr(message, "conversation_id") or message.conversation_id is None:
            if default_conversation_id != -1:
                message.conversation_id = default_conversation_id
            else:
                logger.warning("No valid conversation_id available, using default -1")
                message.conversation_id = -1

        # The message has the right structure already
        # ConversationContext.add_assistant_message will handle extracting the text content
        if (
            message
            and message.content
            and len(message.content) > 0
            and message.content[0].text
        ):
            await conversation_ctx.add_assistant_message(message)
            logger.info(
                f"Assistant message stored for conversation {message.conversation_id}"
            )
        else:
            logger.warning("Empty assistant message content, not storing")
    except Exception as e:  # noqa: BLE001, justified for logging all errors
        logger.error(f"Failed to store assistant message: {e}")


@router.get("/conversations")
async def list_conversations(request: Request):
    """
    List all conversations for the user.
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Check if database is initialized
    if not storage.initialized or storage.conversation is None:
        logger.warning("Database not initialized, cannot list conversations")
        raise HTTPException(status_code=503, detail="Database service unavailable")

    try:
        # Get all conversations for the user
        conversations = await storage.conversation.get_user_conversations(user_id)
        return {"conversations": conversations}
    except Exception as e:  # noqa: BLE001, justified for DB errors
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: int, request: Request):
    """
    Get a specific conversation by ID.
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Check if database is initialized
    if not storage.initialized or storage.conversation is None:
        logger.warning("Database not initialized, cannot get conversation")
        raise HTTPException(status_code=503, detail="Database service unavailable")

    try:
        # Get the conversation
        conversation = await storage.conversation.get_conversation(conversation_id)

        # Check if conversation exists
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Check if user has access to this conversation
        if conversation.user_id != user_id and not is_admin(request):
            raise HTTPException(
                status_code=403, detail="Access denied to this conversation"
            )

        return conversation
    except HTTPException as e:
        raise e
    except Exception as e:  # noqa: BLE001, justified for DB errors
        logger.error(f"Error getting conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e


@router.get("/conversations/{conversation_id}/messages", response_model=list[Message])
async def get_conversation_messages(conversation_id: int, request: Request):
    """
    Get messages for a specific conversation.
    This endpoint retrieves all messages for a given conversation_id and ensures they
    are properly formatted according to the Message schema with valid content arrays.
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Log the request for debugging
    logger.debug(
        f"Getting messages for conversation {conversation_id}, user: {user_id}"
    )

    # Check if database is initialized
    if not storage.initialized or not storage.conversation or not storage.message:
        logger.warning("Database not initialized, cannot get messages")
        raise HTTPException(status_code=503, detail="Database service unavailable")

    try:
        # First check if conversation exists and user has access
        conversation = await storage.conversation.get_conversation(conversation_id)

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Check if user has access to this conversation
        if conversation.user_id != user_id and not is_admin(request):
            raise HTTPException(
                status_code=403, detail="Access denied to this conversation"
            )

        # Get all messages for the conversation
        messages = await storage.message.get_conversation_history(conversation_id)

        # Log retrieved messages for debugging
        logger.debug(
            f"Retrieved {len(messages)} messages for conversation {conversation_id}"
        )

        # Format messages for the response using proper Message objects
        formatted_messages = []
        try:
            for i, msg in enumerate(messages):
                # Log each message for debugging
                logger.debug(
                    f"Processing message {i}: id={msg.id if hasattr(msg, 'id') else 'unknown'}, "
                    + f"role={msg.role if hasattr(msg, 'role') else 'unknown'}, "
                    + f"content_type={type(msg.content) if hasattr(msg, 'content') else 'None'}"
                )

                # Ensure we have a valid message content list
                content_list = []
                if (
                    hasattr(msg, "content")
                    and msg.content
                    and isinstance(msg.content, list)
                    and len(msg.content) > 0
                ):
                    # Use the existing content list
                    content_list = msg.content
                else:
                    # Create a new content list with the text, if available
                    content_text = ""
                    if hasattr(msg, "content"):
                        if msg.content and not isinstance(msg.content, list):
                            # If content is a string, use it
                            content_text = str(msg.content)
                        elif (
                            msg.content
                            and isinstance(msg.content, list)
                            and len(msg.content) > 0
                        ):
                            if hasattr(msg.content[0], "text"):
                                content_text = msg.content[0].text or ""

                    # Create a valid MessageContent object
                    content_list = [
                        MessageContent(
                            type=MessageContentType.TEXT, text=content_text, url=None
                        )
                    ]

                # Get message ID
                msg_id = msg.id if hasattr(msg, "id") else None

                # Get message role
                msg_role = msg.role if hasattr(msg, "role") else MessageRole.USER

                # Get message created_at
                msg_created_at = (
                    msg.created_at
                    if hasattr(msg, "created_at") and msg.created_at
                    else dt.now()
                )

                # Get thinking
                msg_thinking = msg.thinking if hasattr(msg, "thinking") else None

                # Get tool_calls
                msg_tool_calls = msg.tool_calls if hasattr(msg, "tool_calls") else None

                # Create a valid Message object with all required fields
                formatted_messages.append(
                    Message(
                        id=msg_id,
                        role=msg_role,
                        content=content_list,
                        conversation_id=conversation_id,  # Explicitly set conversation_id
                        created_at=msg_created_at,
                        thinking=msg_thinking,
                        tool_calls=msg_tool_calls,
                    )
                )
        except Exception as e:
            logger.error(f"Error formatting messages: {e}")
            # If there's an error in formatting, create a default error message
            formatted_messages = [
                Message(
                    id=1,
                    role=MessageRole.SYSTEM,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=f"Error retrieving messages: {str(e)}",
                        )
                    ],
                    conversation_id=conversation_id,
                    created_at=dt.now(),
                )
            ]

        return formatted_messages
    except HTTPException as e:
        raise e
    except Exception as e:  # noqa: BLE001, justified for DB errors
        logger.error(f"Error fetching messages for conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int, request: Request):
    """
    Delete a conversation and all its messages.
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Check if database is initialized
    if not storage.initialized or storage.conversation is None:
        logger.warning("Database not initialized, cannot delete conversation")
        raise HTTPException(status_code=503, detail="Database service unavailable")

    try:
        # First check if conversation exists and user has access
        db_conversation = await storage.conversation.get_conversation(conversation_id)

        if not db_conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Check if user has access to this conversation
        if db_conversation.user_id != user_id and not is_admin(request):
            raise HTTPException(
                status_code=403, detail="Access denied to this conversation"
            )

        # Delete the conversation
        await storage.conversation.delete_conversation(conversation_id)

        return {
            "status": "success",
            "message": f"Conversation {conversation_id} deleted",
        }
    except HTTPException as e:
        raise e
    except Exception as e:  # noqa: BLE001, justified for DB errors
        logger.error(f"Error deleting conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e


@router.post("/conversations", response_model=Conversation)
async def create_conversation(request: Request):
    """
    Create a new conversation.
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Check if database is initialized
    if not storage.initialized or storage.conversation is None:
        logger.warning("Database not initialized, cannot create conversation")
        raise HTTPException(status_code=503, detail="Database service unavailable")

    try:
        # Create the conversation in the database
        conversation_id = await storage.conversation.create_conversation(
            user_id=user_id
        )

        if not conversation_id:
            raise HTTPException(status_code=500, detail="Failed to create conversation")

        # Get the newly created conversation
        return await storage.conversation.get_conversation(conversation_id)
    except HTTPException as e:
        raise e
    except Exception as e:  # noqa: BLE001, justified for DB errors
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e


@router.post("/pause/{conversation_id}")
async def pause_generation(conversation_id: int, request: Request):
    """
    Pause text generation for a conversation.
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not storage.conversation:
        raise HTTPException(status_code=503, detail="Conversation service unavailable")

    # Check if conversation exists and user has access
    try:
        conversation = await storage.conversation.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if conversation.user_id != user_id and not is_admin(request):
            raise HTTPException(
                status_code=403, detail="Access denied to this conversation"
            )
    except HTTPException as e:
        raise e
    except Exception as e:  # noqa: BLE001, justified for DB errors
        logger.error(f"Error validating conversation access: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e

    # Signal the generation to pause (implementation depends on your streaming setup)
    try:
        # This is a placeholder for actual pause implementation
        # In a real implementation, you might set a flag in a shared state,
        # send a signal to the generator, or use a pub/sub system
        return {"status": "success", "message": "Generation paused"}
    except Exception as e:
        logger.error(f"Error pausing generation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to pause generation: {str(e)}"
        ) from e
