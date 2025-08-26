"""
Conversation Context Manager for RAG functionality.
Handles conversation state, summarization, and retrieval of relevant context.
"""

import logging
from typing import List, Optional, Tuple
from fastapi import HTTPException, Request, status

from runner.pipelines.base_pipeline import Embeddings
from runner.pipelines.factory import pipeline_factory

from models.message import Message
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from models.message_role import MessageRole
from models.user_config import UserConfig
from models.image_metadata import ImageMetadata
from models.conversation import Conversation
from models.conversation_ctx import ConversationCtx
from server.auth import get_request_id, get_user_id
from server.utils.chat.message import extract_message_text
from server.config import logger

from .intent import Intent
from ..db import storage
from .search import SearchContext
from .memory import MemoryContext
from .summary import SummaryContext


class ConversationContext(ConversationCtx):
    """
    Manages context for a conversation, including messages, summaries, and retrieved memories.
    Provides methods for adding messages, summarizing conversations, and retrieving relevant context.
    """

    messages: List[Message]
    notes: List[str]
    images: List[ImageMetadata]
    conversation: Conversation
    current_user_message: Optional[Message] = None
    intent: Optional[Intent] = None

    def __init__(
        self,
        conversation_id: int,
        user_config: UserConfig,
    ):
        """
        Initialize a new conversation context.

        Args:
            user_id: The ID of the user who owns this conversation
            conversation_id: The unique ID of this conversation
            embedding_profile_id: The ID of the embedding model profile to use
            summarization_profile_id: The ID of the summarization model profile to use
            user_config: User configuration settings
        """
        # Core conversation metadata

        # Conversation content
        self.current_user_message = None
        self.notes = []
        self.images = []
        self.intent = Intent()
        self.user_config = user_config

        # Set up logging
        self.logger = logging.getLogger("ConversationContext")

        # Initialize context services
        self.summary_context = SummaryContext(
            user_cfg=user_config, conversation_id=conversation_id
        )

        self.memory_context = MemoryContext(
            user_cfg=user_config, conversation_id=conversation_id
        )

        self.search_context = SearchContext(user_cfg=user_config)

    # -------------------------------------------------------------------------
    # Private Methods - Helper methods for internal use
    # -------------------------------------------------------------------------

    def create_system_prompt(
        self, system_prompt_base: str, dynamic_tool_info: str
    ) -> str:
        """
        Creates a system prompt
        """
        prompt = system_prompt_base or "You are a helpful assistant."

        if dynamic_tool_info:
            prompt += f"\n\nAvailable tools:\n{dynamic_tool_info}\n\nUse these tools strategically to provide comprehensive, accurate responses."

        if self.user_config.memory.enabled:
            prompt += f"\n\nRelevant memories:\n{self.memory_context.memory}"

        if self.user_config.web_search.enabled:
            prompt += f"\n\nResearch findings:\n{self.search_context.research_findings}\n\nUse these findings to support your responses."

        if self.user_config.summarization.enabled:
            prompt += f"\n\nConversation summary:\n{self.summary_context.full_summary}"

        return f"{prompt}\n\nAlways explain your reasoning and cite your sources when using retrieved information."

    async def load_conversation_data(self) -> None:
        """
        Load all conversation data from storage (conversation details, messages, and summaries).
        Should be called after creating a new ConversationContext for an existing conversation.
        """
        # Get conversation details
        try:
            convo = await storage.get_service(storage.conversation).get_conversation(
                self.conversation.id
            )
            if convo:
                self.conversation = convo
        except Exception as e:
            self.logger.error(f"Error loading conversation details: {e}")
            # Continue - we can still work with other data

        # Get messages
        try:
            messages = await storage.get_service(
                storage.message
            ).get_conversation_history(self.conversation.id)
            self.messages = messages or []
        except Exception as e:
            self.logger.error(f"Error loading messages: {e}")
            self.messages = []

    # -------------------------------------------------------------------------
    # Public Methods - Main API for interacting with ConversationContext
    # -------------------------------------------------------------------------

    async def add_message(self, message: Message) -> Tuple[Embeddings, Optional[int]]:
        """
        Add a user message to the conversation, store it in the database,
        discover user intent, update the context, and create embeddings.

        Args:
            message: The user message to add

        Returns:
            A tuple of (embeddings, message_id)
        """
        # Detect intent from the message
        if message.role == MessageRole.USER:
            self.current_user_message = message

            (
                self.intent.detect(message, self.user_config)
                if self.intent is not None
                else None
            )

        # Store message in database
        message_id = await storage.get_service(storage.message).add_message(message)

        # Update message ID and add to messages list
        if message_id:
            message.id = message_id
            self.messages.append(message)

        # Get text for embedding
        text = extract_message_text(message)

        mp = await storage.get_service(storage.model_profile).get_model_profile_by_id(
            self.user_config.model_profiles.embedding_profile_id,
            self.user_config.user_id,
        )
        assert mp, "Embedding model profile not found"

        # Create embeddings using pipeline factory
        embeddings: List[List[float]] = []
        if text:
            try:
                # Get embedding pipeline from factory
                embedding_pipeline = pipeline_factory.get_pipeline(mp)
                return await embedding_pipeline.get([message]), message_id
            except Exception as e:
                self.logger.error(f"Error creating embeddings: {e}")

        return embeddings, message_id

    async def generate_title(self) -> str:
        """
        Generate a conversation title using the formatting model profile

        Returns:
            A generated title for the conversation
        """
        try:
            mp = await storage.get_service(
                storage.model_profile
            ).get_model_profile_by_id(
                self.user_config.model_profiles.formatting_profile_id,
                self.user_config.user_id,
            )

            if not mp:
                raise ValueError("Formatting model profile not found")

            # Get formatting pipeline from factory
            formatting_pipeline = pipeline_factory.get_pipeline(mp)
            # Prepare prompt for title generation
            title_prompt = (
                "Create a short, descriptive title (max 5 words) for this conversation."
            )
            # Convert prompt to a Message for the get method
            format_message = Message(
                role=MessageRole.USER,
                content=[
                    MessageContent(type=MessageContentType.TEXT, text=title_prompt)
                ],
            )
            # Run the pipeline to get formatted title - use get method which is sync
            title = await formatting_pipeline.get([*self.messages, format_message])
            self.conversation.title = title
            return title

        except Exception as e:
            self.logger.error(f"Error generating conversation title: {e}")
            raise

    def clear_notes(self) -> None:
        """Clear all notes"""
        self.notes = []


async def get_conversation_context_from_request(
    request: Request, conversation_id: int
) -> ConversationContext:
    """
    Extract conversation context from the current request.

    Args:
        request: The FastAPI request object
        conversation_id: ID of the conversation to load

    Returns:
        The conversation context for the current request

    Raises:
        HTTPException: If user is not authenticated or conversation is not found
    """
    # Set up request context information
    user_id = get_user_id(request)
    request_id = get_request_id(request)

    if not user_id:
        logger.warning(f"User ID not found for request {request_id}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not authenticated",
        )

    logger.info(f"Processing chat completion request {request_id} for user {user_id}")

    # Fetch user's model profile configuration
    try:
        user_config = await storage.get_service(storage.user_config).get_user_config(
            user_id
        )
        assert user_config, f"User config not found for user {user_id}"
    except AttributeError as e:
        logger.error(f"Error accessing user_config service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="User config service not properly initialized",
        ) from e

    # Initialize conversation context
    conversation_ctx = ConversationContext(
        conversation_id=conversation_id,
        user_config=user_config,
    )

    try:
        # Load the conversation data
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

    return conversation_ctx
