"""
Conversation Context Manager for RAG functionality.
Updated to use pipeline factory instead of direct service instances.
"""

from typing import List, Dict, Optional, Tuple, Any, Union
import os
import sys
from datetime import datetime

# Import model classes
from models.message_content import MessageContent
from models.message import Message
from models.message_role import MessageRole
from models.message_content_type import MessageContentType
from models.summary import Summary
from models.chat_req import ChatReq
from models.user_config import UserConfig

# Import pipeline factory
from runner.pipelines.factory import pipeline_factory

# Add the parent directory to the path to resolve imports
server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if server_dir not in sys.path:
    sys.path.insert(0, server_dir)

# Now we can import server modules
from .intent import Intent, detect_intent
from ..services.search_service import SearchService
from ..db.conversation_storage import ConversationStorage
from ..db.message_storage import MessageStorage
from ..db.summary_storage import SummaryStorage
from ..db.memory_storage import MemoryStorage
from ..db import storage  # Direct access to storage for default instances
import server.config

logger = server.config.logger  # Use the logger from config


class ConversationContext:
    """
    Manages context for a conversation, including messages, summaries, and retrieved memories.
    Updated to use pipeline factory instead of direct service instances.
    """

    def __init__(
        self,
        user_id: str,
        conversation_id: int,
        embedding_profile_id: str,
        summarization_profile_id: str,
        user_config: Optional[UserConfig] = None,
        conversation_storage: Optional[ConversationStorage] = None,
        message_storage: Optional[MessageStorage] = None,
        summary_storage: Optional[SummaryStorage] = None,
        memory_storage: Optional[MemoryStorage] = None,
    ):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.title = ""
        self.master_summary = None
        self.summaries = []
        self.messages = []
        self.retrieved_memories = []
        self.search_results = []
        self.notes = []
        self.images = []
        self.intent = Intent()

        # Store profile IDs for pipeline access
        self.embedding_profile_id = embedding_profile_id
        self.summarization_profile_id = summarization_profile_id
        self.user_config = user_config

        # Storage services - use defaults if not provided
        self._conversation_storage = conversation_storage or storage.conversation
        self._message_storage = message_storage or storage.message
        self._summary_storage = summary_storage or storage.summary
        # Memory storage is passed directly
        self._memory_storage = memory_storage

    def _dict_to_message(self, message_dict: Dict[str, Any]) -> Message:
        """Convert a message dictionary to a Message object"""
        return Message(**message_dict)

    def _dict_to_summary(self, summary_dict: Union[Dict[str, Any], Summary]) -> Summary:
        """Convert a dictionary to a Summary object"""
        if isinstance(summary_dict, Summary):
            return summary_dict
        return Summary(**summary_dict)

    def _extract_title_from_message(self, message: Message) -> str:
        """Extract a title from the first user message"""
        # Default title
        title = "New conversation"

        # Extract title from the first text content if available
        if message.content and len(message.content) > 0:
            for content in message.content:
                if content.type == MessageContentType.TEXT and content.text:
                    # Use the first sentence or truncate if too long
                    text = content.text.strip()
                    if text:
                        # Use first sentence or first 30 chars
                        parts = text.split(".")
                        title = parts[0].strip()
                        if len(title) > 30:
                            title = title[:27] + "..."
                        break

        return title

    def _get_text_for_embedding(self, message: Message) -> str:
        """Extract text content from message for embedding"""
        if not message or not message.content:
            return ""

        text_parts = []
        for content in message.content:
            if content.type == MessageContentType.TEXT and content.text:
                text_parts.append(content.text)

        return " ".join(text_parts)

    async def load_conversation_data(self) -> None:
        """Load conversation data from storage"""
        # Get conversation details
        if self._conversation_storage:
            conversation = await self._conversation_storage.get_conversation(
                self.conversation_id
            )
            if conversation:
                self.title = conversation.title

        # Get messages
        if self._message_storage:
            messages = await self._message_storage.get_conversation_history(
                self.conversation_id
            )
            self.messages = messages or []

        # Get summaries
        if self._summary_storage:
            summaries = await self._summary_storage.get_summaries_for_conversation(
                self.conversation_id
            )
            self.summaries = summaries or []

        # Get master summary if it exists
        # Note: Implementing a placeholder as this functionality is not yet available
        self.master_summary = None

    async def detect_message_intent(self, message: Message) -> None:
        """
        Detect the intent of a user message and update the context's intent.

        Args:
            message: The user message to analyze
        """
        user_config_dict = self.user_config.dict() if self.user_config else None
        self.intent = detect_intent(message, user_config_dict)
        logger.info(f"Detected intent for message: {self.intent.to_dict()}")

    async def add_user_message(
        self, message: Message
    ) -> Tuple[List[List[float]], Optional[int]]:
        """
        Add a user message to the conversation and create embeddings.

        Args:
            message: The user message to add

        Returns:
            A tuple of (embeddings, message_id)
        """
        # Detect intent from the message
        await self.detect_message_intent(message)

        # Store message
        message_id = None
        if (
            self.conversation_id > 0 and self._message_storage
        ):  # Don't store for temporary conversations (-1)
            message_id = await self._message_storage.add_message(
                self.conversation_id,
                message.role.value,
                self._get_text_for_embedding(message),
            )

        # Get text for embedding
        text = self._get_text_for_embedding(message)

        # Create embeddings using pipeline factory
        embeddings: List[List[float]] = []
        if text:
            try:
                # Get embedding pipeline from factory
                embedding_pipeline, _ = pipeline_factory.get_pipeline(
                    self.embedding_profile_id
                )
                if embedding_pipeline:
                    # Convert text to a Message for the get method
                    embed_message = Message(
                        role=MessageRole.USER,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT, text=text, url=None
                            )
                        ],
                        conversation_id=0,
                        tool_calls=None,
                        thinking=None,
                        id=None,
                        created_at=datetime.now(),
                    )
                    # Run the pipeline to get embeddings - use get method which is synchronous
                    embedding_result = embedding_pipeline.get([embed_message])
                    if isinstance(embedding_result, list):
                        if embedding_result and isinstance(embedding_result[0], list):
                            # Already in the right format
                            embeddings = embedding_result
                        else:
                            # Single embedding vector
                            embeddings = [embedding_result]
            except Exception as e:
                logger.error(f"Error creating embeddings: {e}")

        return embeddings, message_id

    async def add_assistant_message(self, message: Message) -> List[List[float]]:
        """
        Add an assistant message to the conversation and create embeddings.

        Args:
            message: The assistant message to add

        Returns:
            List of embedding vectors
        """
        # Store message
        if (
            self.conversation_id > 0 and self._message_storage
        ):  # Don't store for temporary conversations (-1)
            await self._message_storage.add_message(
                self.conversation_id,
                message.role.value,
                self._get_text_for_embedding(message),
            )

        # Get text for embedding
        text = self._get_text_for_embedding(message)

        # Create embeddings using pipeline factory
        embeddings: List[List[float]] = []
        if text:
            try:
                # Get embedding pipeline from factory
                embedding_pipeline, _ = pipeline_factory.get_pipeline(
                    self.embedding_profile_id
                )
                if embedding_pipeline:
                    # Convert text to a Message for the get method
                    embed_message = Message(
                        role=MessageRole.USER,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT, text=text, url=None
                            )
                        ],
                        conversation_id=0,
                        tool_calls=None,
                        thinking=None,
                        id=None,
                        created_at=datetime.now(),
                    )
                    # Run the pipeline to get embeddings - use get method which is synchronous
                    embedding_result = embedding_pipeline.get([embed_message])
                    if isinstance(embedding_result, list):
                        if embedding_result and isinstance(embedding_result[0], list):
                            # Already in the right format
                            embeddings = embedding_result
                        else:
                            # Single embedding vector
                            embeddings = [embedding_result]
            except Exception as e:
                logger.error(f"Error creating embeddings: {e}")

        return embeddings

    async def generate_title_with_formatting(
        self, message: Message, formatting_profile_id: str
    ) -> str:
        """
        Generate a conversation title using the formatting model profile

        Args:
            message: The first user message in the conversation
            formatting_profile_id: The ID of the formatting model profile to use

        Returns:
            A generated title for the conversation
        """
        # Extract text from message
        text = self._get_text_for_embedding(message)
        if not text:
            return "New conversation"

        try:
            # Get formatting pipeline from factory
            formatting_pipeline, _ = pipeline_factory.get_pipeline(
                formatting_profile_id
            )
            if formatting_pipeline:
                # Prepare prompt for title generation
                title_prompt = f"Create a short, descriptive title (max 5 words) for a conversation that begins with this message: '{text}'"

                # Convert prompt to a Message for the get method
                format_message = Message(
                    role=MessageRole.USER,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT, text=title_prompt, url=None
                        )
                    ],
                    conversation_id=0,
                    tool_calls=None,
                    thinking=None,
                    id=None,
                    created_at=datetime.now(),
                )
                # Run the pipeline to get formatted title - use get method which is sync
                title = formatting_pipeline.get([format_message])

                if title:
                    # Clean up and truncate if needed
                    title = title.strip().strip("\"'").strip()
                    if len(title) > 50:
                        title = title[:47] + "..."
                    return title
        except Exception as e:
            logger.error(f"Error generating conversation title: {e}")

        # Fall back to simple extraction if model generation fails
        return self._extract_title_from_message(message)

    def _generate_title(self, message: Message) -> str:
        """Generate a title from the first user message"""
        return self._extract_title_from_message(message)

    def _should_summarize(self) -> bool:
        """Determine if the conversation should be summarized"""
        # Logic to determine if summarization is needed
        # For example, summarize if there are more than 10 messages
        return len(self.messages) >= 10

    def _get_unsummarized_messages(self) -> List[Message]:
        """Get messages that have not been summarized yet"""
        # If there are no summaries, return all messages
        if not self.summaries:
            return self.messages

        # Get the timestamp of the latest summary
        latest_summary_time = max(
            (s.created_at for s in self.summaries if s.created_at is not None),
            default=datetime.min,
        )

        # Return messages created after the latest summary
        return [
            m
            for m in self.messages
            if m.created_at is not None and m.created_at > latest_summary_time
        ]

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

        # Extract text from messages
        texts = []
        for message in unsummarized:
            for content in message.content:
                if content.type == MessageContentType.TEXT and content.text:
                    role_prefix = f"{message.role}: "
                    texts.append(role_prefix + content.text)

        if not texts:
            return None

        combined_text = "\n".join(texts)

        try:
            # Get summarization pipeline from factory
            summarization_pipeline, _ = pipeline_factory.get_pipeline(
                self.summarization_profile_id
            )
            if summarization_pipeline:
                # Convert text to a Message for the get method
                summary_message = Message(
                    role=MessageRole.USER,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT, text=combined_text, url=None
                        )
                    ],
                    conversation_id=0,
                    tool_calls=None,
                    thinking=None,
                    id=None,
                    created_at=datetime.now(),
                )
                # Run the pipeline to get summary - use get method which is synchronous
                summary_text = summarization_pipeline.get([summary_message])

                if summary_text and self._summary_storage:
                    # Get source message IDs for tracking
                    source_ids = [
                        m.id
                        for m in unsummarized
                        if hasattr(m, "id") and m.id is not None
                    ]

                    # Create and store the summary using the appropriate method
                    summary_id = await self._summary_storage.create_summary(
                        self.conversation_id,
                        summary_text,
                        0,  # level 0 for message summaries
                        source_ids,
                    )

                    if summary_id:
                        # Create a Summary object with the returned ID
                        summary = Summary(
                            id=summary_id,
                            conversation_id=self.conversation_id,
                            content=summary_text,
                            level=0,
                            source_ids=source_ids,
                            created_at=datetime.now(),
                        )

                        self.summaries.append(summary)

                        # Check if summaries need to be consolidated
                        await self._check_and_consolidate_summaries()

                        return summary
        except Exception as e:
            logger.error(f"Error summarizing messages: {e}")

        return None

    async def _check_and_consolidate_summaries(self) -> None:
        """Check if summaries need to be consolidated and do so if needed"""
        # Logic to check if we need to consolidate summaries at each level
        level_summaries = {}

        # Group summaries by level
        for summary in self.summaries:
            level = summary.level
            if level not in level_summaries:
                level_summaries[level] = []
            level_summaries[level].append(summary)

        # Check each level for consolidation
        for level, summaries in level_summaries.items():
            if len(summaries) >= 5:  # Consolidate when we have 5+ summaries at a level
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

        # Extract text from summaries
        texts = [s.content for s in level_summaries]
        combined_text = "\n".join(texts)

        try:
            # Get summarization pipeline from factory
            summarization_pipeline, _ = pipeline_factory.get_pipeline(
                self.summarization_profile_id
            )
            if summarization_pipeline and self._summary_storage:
                # Convert text to a Message for the get method
                summary_message = Message(
                    role=MessageRole.USER,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT, text=combined_text, url=None
                        )
                    ],
                    conversation_id=0,
                    tool_calls=None,
                    thinking=None,
                    id=None,
                    created_at=datetime.now(),
                )
                # Run the pipeline to get summary - use get method which is synchronous
                summary_text = summarization_pipeline.get([summary_message])

                if summary_text:
                    # Get source summary IDs for tracking
                    source_ids = [
                        s.id
                        for s in level_summaries
                        if hasattr(s, "id") and s.id is not None
                    ]

                    # Create and store the higher-level summary using the appropriate method
                    summary_id = await self._summary_storage.create_summary(
                        self.conversation_id, summary_text, level + 1, source_ids
                    )

                    if summary_id:
                        # Create a Summary object with the returned ID
                        new_summary = Summary(
                            id=summary_id,
                            conversation_id=self.conversation_id,
                            content=summary_text,
                            level=level + 1,
                            source_ids=source_ids,
                            created_at=datetime.now(),
                        )

                        self.summaries.append(new_summary)

                        # Update master summary if needed
                        await self._update_master_summary()

                        return new_summary
        except Exception as e:
            logger.error(f"Error consolidating summaries: {e}")

        return None

    async def _update_master_summary(self) -> None:
        """Update the master summary if needed"""
        # This method would be implemented once we have master summary functionality
        # For now, it's a placeholder
        logger.debug("Master summary functionality not yet implemented")

    async def _create_master_summary(self) -> None:
        """Create a new master summary"""
        # This method would be implemented once we have master summary functionality
        # For now, it's a placeholder
        logger.debug("Master summary creation functionality not yet implemented")

    async def _update_existing_master_summary(self) -> None:
        """Update the existing master summary with new information"""
        # This method would be implemented once we have master summary functionality
        # For now, it's a placeholder
        logger.debug("Master summary update functionality not yet implemented")

    async def retrieve_memories(self, query: str) -> List[Any]:
        """
        Retrieve relevant memories for the query.
        Only called when intent.memory is True.

        Args:
            query: The query text

        Returns:
            List of retrieved memories
        """
        if not self.intent.memory or not query or not self._memory_storage:
            return []

        try:
            # Get embedding pipeline from factory
            embedding_pipeline, _ = pipeline_factory.get_pipeline(
                self.embedding_profile_id
            )
            if embedding_pipeline:
                # Convert query to a Message for the get method
                query_message = Message(
                    role=MessageRole.USER,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT, text=query, url=None
                        )
                    ],
                    conversation_id=0,
                    tool_calls=None,
                    thinking=None,
                    id=None,
                    created_at=datetime.now(),
                )
                # Run the pipeline to get embeddings - use get method which is synchronous
                embedding_result = embedding_pipeline.get([query_message])

                if embedding_result and isinstance(embedding_result, list):
                    # Ensure we have the correct format for embeddings
                    embeddings: List[List[float]] = []
                    if isinstance(embedding_result[0], list):
                        # Already in the right format (list of lists)
                        embeddings = embedding_result
                    else:
                        # Single embedding vector, convert to list of lists
                        embeddings = [embedding_result]

                    # Search for memories with the embedding
                    memories = await self._memory_storage.search_similarity(
                        embeddings,
                        min_similarity=0.7,
                        limit=5,
                        user_id=self.user_id,
                    )
                    self.retrieved_memories = memories
                    return memories
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")

        return []

    async def search_web(self, query: str) -> List[Any]:
        """
        Search the web for the query.
        Only called when intent.web_search is True.

        Args:
            query: The query text

        Returns:
            List of search results
        """
        if not self.intent.web_search or not query:
            return []

        try:
            # Only instantiate SearchService when needed based on intent
            search_service = SearchService()
            results = await search_service.search(query)
            self.search_results = results
            return list(results) if hasattr(results, "__iter__") else [results]
        except Exception as e:
            logger.error(f"Error searching web: {e}")

        return []

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

        # Find the last user message
        for i in range(len(request.messages) - 1, -1, -1):
            if request.messages[i].role == MessageRole.USER:
                return request.messages[i]

        return None
