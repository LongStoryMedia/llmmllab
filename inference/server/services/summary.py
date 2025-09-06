"""
Conversation summarization functionality for RAG system.
"""

from typing import Dict, List, Optional
from datetime import datetime, timezone

from models import (
    Message,
    Summary,
    UserConfig,
    MessageRole,
    MessageContent,
    MessageContentType,
    ChatResponse,
)
from server.db import storage
from server.config import logger

from runner import pipeline_factory
from utils.message import extract_message_text


class SummaryContext:
    """
    Context for performing conversation summarization and consolidation.
    Manages the list of summaries and the master summary for a conversation.
    """

    summaries: List[Summary]
    master_summary: Optional[Summary]
    is_summarized: bool
    full_summary: str

    def __init__(self, user_cfg: UserConfig, conversation_id: int):
        """
        Initialize the conversation summarization context.

        Args:
            user_cfg: The user configuration
            conversation_id: The ID of the conversation
        """
        self.user_config = user_cfg
        self.user_id = user_cfg.user_id
        self.conversation_id = conversation_id
        self.logger = logger
        self.summaries = []
        self.master_summary = None
        self.is_summarized = False
        self.full_summary = ""

    async def load_summaries(self) -> None:
        """
        Load all summaries for the conversation from storage.
        """
        try:
            self.summaries = await storage.get_service(
                storage.summary
            ).get_summaries_for_conversation(self.conversation_id)
            # Find the highest-level summary as the master summary
            if self.summaries:
                try:
                    max_level = max(s.level for s in self.summaries)
                    master_summaries = [
                        s for s in self.summaries if s.level == max_level
                    ]
                    if master_summaries:
                        self.master_summary = max(
                            master_summaries,
                            key=lambda s: self._aware(s.created_at),
                        )
                except Exception as e:
                    self.logger.error(f"Error finding master summary: {e}")
                    self.master_summary = None
            self.logger.debug(f"Loaded {len(self.summaries)} summaries")
        except Exception as e:
            self.logger.error(f"Error loading summaries: {e}")
            self.summaries = []
            self.master_summary = None

    def _should_summarize(self, messages: List[Message]) -> bool:
        """
        Determine if the conversation should be summarized.

        Args:
            messages: List of conversation messages

        Returns:
            True if messages should be summarized, False otherwise
        """
        return len(messages) >= self.user_config.summarization.messages_before_summary

    def _aware(self, dt: Optional[datetime]) -> datetime:
        """Return a timezone-aware UTC datetime for comparisons/sorts.

        - If dt is None, return datetime.min in UTC.
        - If dt is naive, assume UTC and attach tzinfo=UTC.
        - If dt is aware, convert to UTC.
        """
        if dt is None:
            return datetime.min.replace(tzinfo=timezone.utc)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _get_unsummarized_messages(
        self, messages: List[Message], summaries: List[Summary]
    ) -> List[Message]:
        """
        Get messages that have not been summarized yet.

        Args:
            messages: List of all conversation messages
            summaries: List of all summaries

        Returns:
            Messages created after the most recent level 1 summary,
            ordered chronologically from oldest to newest, limited to the number
            specified in the user's summarization config.
        """
        # Get the limit from user configuration if available
        messages_limit = self.user_config.summarization.messages_before_summary

        # Find level 1 summaries - if none exist, we'll use all messages
        level_1_summaries = [s for s in summaries if s.level == 1]

        # Filter messages based on summaries
        if not level_1_summaries:
            # No level 1 summaries exist, use all messages
            unsummarized_messages = messages
        else:
            # Find the timestamp of the most recent level 1 summary
            most_recent_summary = max(
                level_1_summaries,
                key=lambda s: self._aware(s.created_at),
            )
            most_recent_timestamp = self._aware(most_recent_summary.created_at)

            # Get all messages created after the most recent level 1 summary
            unsummarized_messages = [
                m
                for m in messages
                if m.created_at is not None
                and self._aware(m.created_at) > most_recent_timestamp
            ]

        # Sort them by creation time (oldest first)
        unsummarized_messages.sort(key=lambda m: self._aware(m.created_at))

        # Return unsummarized messages limited by config
        return unsummarized_messages[:messages_limit]

    async def _create_summary(
        self,
        messages: List[Message],
    ) -> Optional[Summary]:
        """
        Create a summary from the provided messages.

        Args:
            messages: Messages to summarize
            summarization_profile_id: The ID of the summarization model profile

        Returns:
            New summary if created, None otherwise
        """
        if not messages:
            return None

        try:
            mp = await storage.get_service(
                storage.model_profile
            ).get_model_profile_by_id(
                self.user_config.model_profiles.summarization_profile_id,
                self.user_config.user_id,
            )
            assert mp

            with pipeline_factory.pipeline(mp, str) as pipe:
                # Run the pipeline to get summary - use get method which is synchronous
                summary_text = await pipe.process_messages(messages)
            # Coerce ChatResponse to text if pipeline returned ChatResponse
            if isinstance(summary_text, ChatResponse):
                summary_text = (
                    extract_message_text(summary_text.message)
                    if summary_text.message
                    else ""
                )
            if summary_text:
                # Get source message IDs for tracking
                source_ids = [m.id for m in messages if m.id is not None]
                # Create a Summary object
                summary = Summary(
                    id=-1,
                    conversation_id=self.conversation_id,
                    content=summary_text,
                    level=1,
                    source_ids=source_ids,
                    created_at=datetime.now(timezone.utc),
                )
                # Create and store the summary using the appropriate method
                summary_id = await storage.get_service(storage.summary).create_summary(
                    summary
                )
                if summary_id:
                    summary.id = summary_id  # Update ID with the stored one
                    return summary
        except Exception as e:
            self.logger.error(f"Error summarizing messages: {e}")

        return None

    async def check_and_consolidate_summaries(
        self, summaries: List[Summary], master_summary: Optional[Summary] = None
    ) -> Optional[Summary]:
        """
        Check if summaries need to be consolidated and do so if needed.

        Args:
            summaries: List of all existing summaries
            master_summary: Current master summary, if any

        Returns:
            Updated master summary if changed, None otherwise
        """
        # Logic to check if we need to consolidate summaries at each level
        level_summaries: Dict[int, List[Summary]] = {}

        # Get configuration values from user config
        consolidation_threshold = (
            self.user_config.summarization.summaries_before_consolidation
        )
        max_summary_levels = self.user_config.summarization.max_summary_levels

        # Group summaries by level
        for summary in summaries:
            level = summary.level
            if level not in level_summaries:
                level_summaries[level] = []
            level_summaries[level].append(summary)

        # New master summary if created during consolidation
        new_master_summary = None

        # Check each level for consolidation
        for level, level_summaries_list in level_summaries.items():
            if len(level_summaries_list) >= consolidation_threshold:
                if level == max_summary_levels:
                    # For max level summaries, consolidate into master summary
                    new_master_summary = await self._create_or_update_master_summary(
                        summaries,
                        master_summary,
                        max_summary_levels,
                        consolidation_threshold,
                    )
                else:
                    # For lower levels, create a higher level summary
                    await self._consolidate_level(level, summaries)

        return new_master_summary

    async def _consolidate_level(
        self, level: int, summaries: List[Summary]
    ) -> Optional[Summary]:
        """
        Consolidate summaries at a specific level into a higher-level summary.

        Args:
            level: The level to consolidate
            summaries: List of all summaries

        Returns:
            New consolidated summary if created, None otherwise
        """
        # Get summaries at this level
        level_summaries = [s for s in summaries if s.level == level]
        if len(level_summaries) < 2:
            return None

        # Sort by creation time
        level_summaries.sort(key=lambda s: self._aware(s.created_at))

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
            with pipeline_factory.pipeline(profile, str) as pipe:
                summary_text = await pipe.process_messages(msgs)

                if isinstance(summary_text, ChatResponse):
                    summary_text = (
                        extract_message_text(summary_text.message)
                        if summary_text.message
                        else ""
                    )

                if summary_text:
                    # Get source summary IDs for tracking
                    source_ids = [s.id for s in level_summaries]
                    new_summary = Summary(
                        id=-1,
                        conversation_id=self.conversation_id,
                        content=summary_text,
                        level=level + 1,
                        source_ids=source_ids,
                        created_at=datetime.now(timezone.utc),
                    )

                    # Create and store the higher-level summary
                    summary_id = await storage.get_service(
                        storage.summary
                    ).create_summary(new_summary)

                    if summary_id:
                        # Update with the returned ID
                        new_summary.id = summary_id
                        return new_summary
        except Exception as e:
            self.logger.error(f"Error consolidating summaries: {e}")

        return None

    async def summarize(self, messages: List[Message]):
        """
        Main entry point for summarization workflow. Handles creating summaries
        and consolidating them as needed.

        Args:
            messages: The list of messages to potentially summarize
            summarization_profile_id: ID of the profile to use for summarization\
        """
        # Load summaries if not already loaded
        if not self.summaries:
            await self.load_summaries()

        # First check if we need to create a new summary
        new_summary = None
        if self._should_summarize(messages):
            unsummarized = self._get_unsummarized_messages(messages, self.summaries)
            if unsummarized:
                new_summary = await self._create_summary(unsummarized)
                if new_summary:
                    self.summaries.append(new_summary)

        # Then check if we need to consolidate summaries
        new_master = await self.check_and_consolidate_summaries(
            self.summaries, self.master_summary
        )
        if new_master:
            self.master_summary = new_master

        self._set_full_summary()
        self.is_summarized = True

    def _set_full_summary(self):
        """
        Concatenates all summaries into a single string
        """
        if not self.summaries:
            return ""

        # Sort summaries by creation time
        self.summaries.sort(key=lambda s: self._aware(s.created_at))

        # Concatenate all summary texts
        self.full_summary = "\n\n".join(s.content for s in self.summaries)

    async def _create_or_update_master_summary(
        self,
        summaries: List[Summary],
        master_summary: Optional[Summary],
        max_level: int,
        consolidation_threshold: int,
    ) -> Optional[Summary]:
        """
        Create or update the master summary by consolidating summaries at max level.

        Args:
            summaries: List of all summaries
            master_summary: Current master summary, if any
            max_level: Maximum summary level from config
            consolidation_threshold: Threshold for consolidation from config

        Returns:
            Updated or new master summary if created/updated, None otherwise
        """
        # Check if we have enough summaries at max level to consolidate
        max_level_summaries = [s for s in summaries if s.level == max_level]

        if len(max_level_summaries) < consolidation_threshold:
            # Not enough summaries at max level
            return None

        # Sort max level summaries by creation time
        max_level_summaries.sort(key=lambda s: self._aware(s.created_at))

        # Determine which summaries to consolidate
        summaries_to_consolidate = max_level_summaries

        # If a master summary already exists, include it
        if master_summary:
            summaries_to_consolidate = max_level_summaries + [master_summary]
            # Re-sort including the master summary
            summaries_to_consolidate.sort(key=lambda s: self._aware(s.created_at))

        try:
            profile = await storage.get_service(
                storage.model_profile
            ).get_model_profile_by_id(
                self.user_config.model_profiles.summarization_profile_id, self.user_id
            )
            assert profile, "Failed to retrieve model profile"
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

            # Get summarization pipeline from factory
            with pipeline_factory.pipeline(profile, str) as pipe:
                # Run the pipeline to get summary
                summary_text = await pipe.process_messages(summary_messages)
                if isinstance(summary_text, ChatResponse):
                    summary_text = (
                        extract_message_text(summary_text.message)
                        if summary_text.message
                        else ""
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
                        created_at=datetime.now(timezone.utc),
                    )

                    # Create and store the master summary
                    summary_id = await storage.get_service(
                        storage.summary
                    ).create_summary(new_summary)

                    if summary_id:
                        # Update with the returned ID
                        new_summary.id = summary_id
                        return new_summary
        except Exception as e:
            self.logger.error(f"Error creating/updating master summary: {e}")

        return None
