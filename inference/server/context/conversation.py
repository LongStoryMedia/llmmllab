"""
Conversation Context Manager for RAG functionality.
Handles conversation state, summarization, and retrieval of relevant context.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, cast
from fastapi import HTTPException, Request, status

from runner.pipelines.base_pipeline import Embeddings
from runner.pipelines.factory import pipeline_factory

from models.search_topic_synthesis import SearchTopicSynthesis
from models.chat_req import ChatReq
from models.message import Message
from models.memory import Memory
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from models.message_role import MessageRole
from models.summary import Summary
from models.user_config import UserConfig
from models.search_result import SearchResult
from models.image_metadata import ImageMetadata

import server.config
from server.auth import get_request_id, get_user_id, is_admin
from server.utils.chat.message import extract_message_text

from .intent import Intent, detect_intent
from ..db import storage
from ..services.search_service import SearchService

logger = server.config.logger  # Use the logger from config


class ConversationContext:
    """
    Manages context for a conversation, including messages, summaries, and retrieved memories.
    Provides methods for adding messages, summarizing conversations, and retrieving relevant context.
    """

    summaries: List[Summary]
    master_summary: Optional[Summary]
    messages: List[Message]
    retrieved_memories: List[Memory]
    search_results: List[SearchTopicSynthesis]
    notes: List[str]
    images: List[ImageMetadata]
    user_id: str
    conversation_id: int
    current_user_message: Optional[Message]

    def __init__(
        self,
        user_id: str,
        conversation_id: int,
        embedding_profile_id: str,
        summarization_profile_id: str,
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
            conversation_storage: Optional custom conversation storage service
            message_storage: Optional custom message storage service
            summary_storage: Optional custom summary storage service
            memory_storage: Optional custom memory storage service
        """
        # Core conversation metadata
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.title = ""

        # Conversation content
        self.master_summary = None
        self.current_user_message = None
        self.summaries = []
        self.messages = []
        self.retrieved_memories = []
        self.search_results = []
        self.notes = []
        self.images = []
        self.intent = Intent()

        # Model configuration
        self.embedding_profile_id = embedding_profile_id
        self.summarization_profile_id = summarization_profile_id
        self.user_config = user_config

        # Set up logging
        self.logger = server.config.logger
        self.logger.name = "ConversationContext"

    # -------------------------------------------------------------------------
    # Private Methods - Helper methods for internal use
    # -------------------------------------------------------------------------

    async def _load_conversation_data(self) -> None:
        """
        Load all conversation data from storage (conversation details, messages, and summaries).
        Should be called after creating a new ConversationContext for an existing conversation.
        """
        # Get conversation details
        try:
            conversation = await storage.get_service(
                storage.conversation
            ).get_conversation(self.conversation_id)
            if conversation:
                self.title = conversation.title
        except Exception as e:
            self.logger.error(f"Error loading conversation details: {e}")
            # Continue - we can still work with other data

        # Get messages
        try:
            messages = await storage.get_service(
                storage.message
            ).get_conversation_history(self.conversation_id)
            self.messages = messages or []
        except Exception as e:
            self.logger.error(f"Error loading messages: {e}")
            self.messages = []

        # Get summaries
        try:
            summaries = await storage.get_service(
                storage.summary
            ).get_summaries_for_conversation(self.conversation_id)
            self.summaries = summaries or []
        except Exception as e:
            self.logger.error(f"Error loading summaries: {e}")
            self.summaries = []

        # Get master summary if it exists
        # Find the highest-level summary as the master summary
        if self.summaries:
            try:
                max_level = max(s.level for s in self.summaries)
                master_summaries = [s for s in self.summaries if s.level == max_level]
                if master_summaries:
                    self.master_summary = max(
                        master_summaries,
                        key=lambda s: s.created_at if s.created_at else datetime.min,
                    )
            except Exception as e:
                self.logger.error(f"Error finding master summary: {e}")
                self.master_summary = None

    # -------------------------------------------------------------------------
    # Public Methods - Main API for interacting with ConversationContext
    # -------------------------------------------------------------------------

    async def add_user_message(
        self, message: Message
    ) -> Tuple[Embeddings, Optional[int]]:
        """
        Add a user message to the conversation, store it in the database,
        discover user intent, update the context, and create embeddings.

        Args:
            message: The user message to add

        Returns:
            A tuple of (embeddings, message_id)
        """
        # Detect intent from the message
        self.intent = detect_intent(message, self.user_config)

        # Store message in database
        message_id = await storage.get_service(storage.message).add_message(
            self.conversation_id,
            message.role.value,
            message.content,
        )

        # Update message ID and add to messages list
        if message_id:
            message.id = message_id
            self.messages.append(message)

        # Get text for embedding
        text = extract_message_text(message)

        # Create embeddings using pipeline factory
        embeddings: List[List[float]] = []
        if text:
            try:
                # Get embedding pipeline from factory
                embedding_pipeline, _ = pipeline_factory.get_pipeline(
                    self.embedding_profile_id
                )
                return await embedding_pipeline.emb(text, True, 768), message_id
            except Exception as e:
                self.logger.error(f"Error creating embeddings: {e}")

        return embeddings, message_id

    async def add_assistant_message(self, message: Message) -> Embeddings:
        """
        Add an assistant message to the conversation, store it in the database,
        update the context, and create embeddings.

        Args:
            message: The assistant message to add

        Returns:
            List of embedding vectors
        """
        # Store message in database
        message_id = await storage.get_service(storage.message).add_message(
            self.conversation_id,
            message.role.value,
            message.content,
        )

        # Update message ID and add to messages list
        if message_id:
            message.id = message_id
            self.messages.append(message)

        # Get text for embedding
        text = extract_message_text(message)
        # Create embeddings using pipeline factory
        embeddings: List[List[float]] = []
        if text:
            try:
                # Get embedding pipeline from factory
                embedding_pipeline, _ = pipeline_factory.get_pipeline(
                    self.embedding_profile_id
                )
                return await embedding_pipeline.emb(text, False, 768)
            except Exception as e:
                self.logger.error(f"Error creating embeddings: {e}")

        return embeddings

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
            formatting_pipeline, _ = pipeline_factory.get_pipeline(mp.name)
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
            return formatting_pipeline.get([*self.messages, format_message])
        except Exception as e:
            self.logger.error(f"Error generating conversation title: {e}")
            raise

    def _should_summarize(self) -> bool:
        """Determine if the conversation should be summarized"""
        return (
            len(self.messages) >= self.user_config.summarization.messages_before_summary
        )

    def _get_unsummarized_messages(self) -> List[Message]:
        """
        Get messages that have not been summarized yet.

        Returns messages created after the most recent level 1 summary,
        ordered chronologically from oldest to newest, limited to the number
        specified in the user's summarization config.
        """
        # Get the limit from user configuration if available
        messages_limit = 10  # Default value
        if self.user_config and self.user_config.summarization:
            messages_limit = self.user_config.summarization.messages_before_summary

        # Find level 1 summaries - if none exist, we'll use all messages
        level_1_summaries = [s for s in self.summaries if s.level == 1]

        # Filter messages based on summaries
        if not level_1_summaries:
            # No level 1 summaries exist, use all messages
            unsummarized_messages = self.messages
        else:
            # Find the timestamp of the most recent level 1 summary
            most_recent_summary = max(
                level_1_summaries,
                key=lambda s: (
                    s.created_at if s.created_at is not None else datetime.min
                ),
            )
            most_recent_timestamp = most_recent_summary.created_at

            # Get all messages created after the most recent level 1 summary
            unsummarized_messages = [
                m
                for m in self.messages
                if m.created_at is not None and m.created_at > most_recent_timestamp
            ]

        # Now that we have filtered the messages, sort them by creation time (oldest first)
        unsummarized_messages.sort(
            key=lambda m: m.created_at if m.created_at is not None else datetime.min
        )

        # Return unsummarized messages limited by config
        return unsummarized_messages[:messages_limit]

    async def summarize_messages(self) -> Optional[Summary]:
        """
        Summarize unsummarized messages if needed.

        Returns:
            New summary if created, None otherwise
        """
        if not self._should_summarize():
            return None

        unsummarized = self._get_unsummarized_messages()
        if not unsummarized:
            return None

        try:
            # Get summarization pipeline from factory
            summarization_pipeline, _ = pipeline_factory.get_pipeline(
                self.summarization_profile_id
            )
            # Run the pipeline to get summary - use get method which is synchronous
            summary_text = summarization_pipeline.get(unsummarized)
            if summary_text:
                # Get source message IDs for tracking
                source_ids = [m.id for m in unsummarized if m.id is not None]
                # Create a Summary object with the returned ID
                summary = Summary(
                    id=-1,
                    conversation_id=self.conversation_id,
                    content=summary_text,
                    level=1,
                    source_ids=source_ids,
                    created_at=datetime.now(),
                )
                # Create and store the summary using the appropriate method
                summary_id = await storage.get_service(storage.summary).create_summary(
                    summary
                )
                if summary_id:
                    summary.id = summary_id  # Update ID with the stored one
                    self.summaries.append(summary)
                    # Check if summaries need to be consolidated
                    await self._check_and_consolidate_summaries()
                    return summary
        except Exception as e:
            self.logger.error(f"Error summarizing messages: {e}")

        return None

    async def _check_and_consolidate_summaries(self) -> None:
        """Check if summaries need to be consolidated and do so if needed"""
        # Logic to check if we need to consolidate summaries at each level
        level_summaries: Dict[int, List[Summary]] = {}

        # Get configuration values from user config
        consolidation_threshold = (
            self.user_config.summarization.summaries_before_consolidation
        )
        max_summary_levels = self.user_config.summarization.max_summary_levels

        # Group summaries by level
        for summary in self.summaries:
            level = summary.level
            if level not in level_summaries:
                level_summaries[level] = []
            level_summaries[level].append(summary)

        # Check each level for consolidation
        for level, summaries in level_summaries.items():
            if (
                len(summaries) >= consolidation_threshold
            ):  # Consolidate based on user config
                if level == max_summary_levels:
                    # For max level summaries, consolidate into master summary
                    await self._create_or_update_master_summary()
                else:
                    # For lower levels, create a higher level summary
                    await self._consolidate_level(level)

    async def _consolidate_level(self, level: int) -> Optional[Summary]:
        """
        Consolidate summaries at a specific level into a higher-level summary.

        Args:
            level: The level to consolidate

        Returns:
            New consolidated summary if created, None otherwise
        """
        # Get summaries at this level
        level_summaries = [s for s in self.summaries if s.level == level]
        if len(level_summaries) < 2:
            return None

        # Sort by creation time
        level_summaries.sort(key=lambda s: s.created_at)

        msgs = [
            Message(
                role=MessageRole.SYSTEM,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT, text=s.content, url=None
                    )
                ],
            )
            for s in level_summaries
        ]

        try:
            profile = await storage.get_service(
                storage.model_profile
            ).get_model_profile_by_id(
                self.user_config.model_profiles.summarization_profile_id, self.user_id
            )
            assert profile, "Failed to retrieve model profile"
            # Get summarization pipeline from factory
            summarization_pipeline, _ = pipeline_factory.get_pipeline(profile.name)
            if summarization_pipeline:
                # Run the pipeline to get summary - use get method which is synchronous
                summary_text = summarization_pipeline.get(msgs, profile.parameters)

                if summary_text:
                    # Get source summary IDs for tracking
                    source_ids = [s.id for s in level_summaries]
                    new_summary = Summary(
                        id=-1,
                        conversation_id=self.conversation_id,
                        content=summary_text,
                        level=level + 1,
                        source_ids=source_ids,
                        created_at=datetime.now(),
                    )

                    # Create and store the higher-level summary using the appropriate method
                    summary_id = await storage.get_service(
                        storage.summary
                    ).create_summary(new_summary)

                    if summary_id:
                        # Create a Summary object with the returned ID
                        new_summary.id = summary_id

                        # Remove the consolidated summaries from self.summaries
                        self.summaries = [
                            s for s in self.summaries if s.id not in source_ids
                        ]
                        # Add the new summary
                        self.summaries.append(new_summary)

                        # Update master summary if needed
                        await self._create_or_update_master_summary()

                        return new_summary
        except Exception as e:
            self.logger.error(f"Error consolidating summaries: {e}")

        return None

    async def _create_or_update_master_summary(self) -> Optional[Summary]:
        """
        Create or update the master summary by consolidating summaries at max level

        Returns:
            Updated or new master summary if created/updated, None otherwise
        """
        # Get max level from user config
        max_level = 3  # Default value
        consolidation_threshold = 3  # Default value
        if self.user_config and self.user_config.summarization:
            max_level = self.user_config.summarization.max_summary_levels
            consolidation_threshold = (
                self.user_config.summarization.summaries_before_consolidation
            )

        # Check if we have enough summaries at max level to consolidate
        max_level_summaries = [s for s in self.summaries if s.level == max_level]

        if len(max_level_summaries) < consolidation_threshold:
            # Not enough summaries at max level
            return None

        # Sort max level summaries by creation time
        max_level_summaries.sort(key=lambda s: s.created_at)

        # Determine which summaries to consolidate
        summaries_to_consolidate = max_level_summaries

        # If a master summary already exists, include it
        if self.master_summary:
            summaries_to_consolidate = max_level_summaries + [self.master_summary]
            # Re-sort including the master summary
            summaries_to_consolidate.sort(key=lambda s: s.created_at)

        try:
            profile = await storage.get_service(
                storage.model_profile
            ).get_model_profile_by_id(
                self.user_config.model_profiles.summarization_profile_id, self.user_id
            )
            assert profile, "Failed to retrieve model profile"
            # Get summarization pipeline from factory
            summarization_pipeline, _ = pipeline_factory.get_pipeline(profile.name)
            if summarization_pipeline:
                # Convert each summary to a Message object
                summary_messages = []
                for summary in summaries_to_consolidate:
                    summary_message = Message(
                        role=MessageRole.USER,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT,
                                text=summary.content,
                                url=None,
                            )
                        ],
                    )
                    summary_messages.append(summary_message)

                # Run the pipeline to get summary
                summary_text = summarization_pipeline.get(
                    summary_messages, profile.parameters
                )

                if summary_text:
                    # Get source summary IDs for tracking
                    source_ids = [s.id for s in summaries_to_consolidate]

                    # Create updated master summary with special level (max_level + 1)
                    new_summary = Summary(
                        id=-1,
                        conversation_id=self.conversation_id,
                        content=summary_text,
                        level=max_level + 1,  # Special level for master summary
                        source_ids=source_ids,
                        created_at=datetime.now(),
                    )

                    # Create and store the master summary
                    summary_id = await storage.get_service(
                        storage.summary
                    ).create_summary(new_summary)

                    if summary_id:
                        # Create a Summary object with the returned ID
                        new_summary.id = summary_id
                        self.master_summary = new_summary

                        # Remove the consolidated summaries from self.summaries
                        self.summaries = [
                            s for s in self.summaries if s.id not in source_ids
                        ]

                        # Also add to summaries list
                        self.summaries.append(new_summary)

                        return new_summary
        except Exception as e:
            self.logger.error(f"Error creating/updating master summary: {e}")

        return None

    async def retrieve_memories(
        self,
        embeddings: Embeddings,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Memory]:
        """
        Retrieve relevant memories for the query.
        Only called when intent.memory is True.

        Args:
            query: The query text

        Returns:
            List of retrieved memories
        """
        if not self.intent.memory:
            return []
        if self.retrieved_memories:
            return self.retrieved_memories

        try:
            # Search for memories with the embedding
            memories = await storage.get_service(storage.memory).search_similarity(
                embeddings,
                min_similarity=self.user_config.memory.similarity_threshold,
                limit=self.user_config.memory.limit,
                user_id=(
                    self.user_id
                    if not self.user_config.memory.enable_cross_user
                    else None
                ),
                conversation_id=(
                    self.conversation_id
                    if not self.user_config.memory.enable_cross_conversation
                    else None
                ),
                start_date=start_date,
                end_date=end_date,
            )
            self.retrieved_memories = memories
            return memories
        except Exception as e:
            self.logger.error(f"Error retrieving memories: {e}")

        return []

    async def search_web(self, message: Message) -> List[SearchTopicSynthesis]:
        """
        Search the web for the query.
        Only called when intent.web_search is True.

        Args:
            message: The user message containing the query

        Returns:
            List of search results
        """
        if not self.intent.web_search:
            raise ValueError("Web search intent is not enabled.")
        if self.search_results:
            return self.search_results

        try:
            # Only instantiate SearchService when needed based on intent
            search_service = SearchService(self.user_config)
            self.search_results = await search_service.search(
                message, self.conversation_id
            )
            return self.search_results
        except Exception as e:
            self.logger.error(f"Error searching web: {e}")

        raise ValueError("Web search failed.")

    def clear_notes(self) -> None:
        """Clear all notes"""
        self.notes = []

    def get_current_user_message(self, request: ChatReq) -> Optional[Message]:
        """
        Get the current user message from a chat request.

        Args:
            request: The chat request

        Returns:
            The current user message if found, None otherwise
        """
        if not request or not request.messages:
            return None
        if self.current_user_message:
            return self.current_user_message

        # Find the last user message
        for i in range(len(request.messages) - 1, -1, -1):
            if request.messages[i].role == MessageRole.USER:
                self.current_user_message = request.messages[i]
                return request.messages[i]

        return None


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

    # Validate storage services are available
    required_services = [
        storage.user_config,
        storage.conversation,
        storage.model_profile,
        storage.message,
        storage.memory,
        storage.summary,
    ]

    missing_services = [service for service in required_services if service is None]
    if missing_services:
        missing_names = ", ".join(missing_services)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Required database services not initialized: {missing_names}",
        )

    # Fetch user's model profile configuration
    try:
        user_config = await storage.get_service(storage.user_config).get_user_config(
            user_id
        )
        if not user_config or not user_config.model_profiles:
            logger.warning(f"No model profile configuration found for user {user_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User model profile configuration not found",
            )
    except AttributeError as e:
        logger.error(f"Error accessing user_config service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="User config service not properly initialized",
        ) from e

    # Get model profiles from user config
    model_profiles = user_config.model_profiles

    # Get profile IDs from user config
    embedding_profile_id = str(model_profiles.embedding_profile_id)
    summarization_profile_id = str(model_profiles.summarization_profile_id)

    logger.info(f"Using embedding profile: {embedding_profile_id}")
    logger.info(f"Using summarization profile: {summarization_profile_id}")

    # Verify conversation exists and user has access
    try:
        conversation = await storage.get_service(storage.conversation).get_conversation(
            conversation_id
        )
        if not conversation or (
            conversation.user_id != user_id and not is_admin(request)
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this conversation",
            )
    except AttributeError as e:
        logger.error(f"Error accessing conversation service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation service not properly initialized",
        ) from e

    # Initialize conversation context
    conversation_ctx = ConversationContext(
        user_id=user_id,
        conversation_id=conversation_id,
        embedding_profile_id=embedding_profile_id,
        summarization_profile_id=summarization_profile_id,
        user_config=user_config,
    )

    try:
        await conversation_ctx._load_conversation_data()
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
