"""
Direct port of Maistro's conversation.go storage logic to Python with cache integration.
"""

from typing import List, Optional
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from models.conversation import Conversation
from db.cache_storage import cache_storage
from utils.logging import llmmllogger
from .userconfig_storage import UserConfigStorage

logger = llmmllogger.bind(component="conversation_storage")


class ConversationStorage:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        user_config_storage: UserConfigStorage,
    ):
        self.session_factory = session_factory
        self.user_config_storage = user_config_storage  # Will be set by Storage class

    async def create_conversation(self, conversation: Conversation) -> Optional[int]:
        async with self.session_factory() as session:
            # Ensure the user exists with proper default config before creating the conversation
            if self.user_config_storage:
                await self.user_config_storage.ensure_user_exists(conversation.user_id)
            else:
                # Fallback to old method if UserConfigStorage not available
                await session.execute(
                    text("INSERT INTO users(id) VALUES (:user_id) ON CONFLICT (id) DO NOTHING"),
                    {"user_id": conversation.user_id},
                )

            result = await session.execute(
                text(
                    "INSERT INTO conversations(user_id, title) VALUES (:user_id, :title)"
                    " RETURNING id, user_id, title, created_at, updated_at"
                ),
                {"user_id": conversation.user_id, "title": conversation.title},
            )
            row = result.one()
            assert row
            conversation_id = row[0] if row else None

            # Cache the new conversation if successful
            if conversation_id:
                row_dict = dict(row._mapping)
                conversation = Conversation(**row_dict)
                await session.commit()
                cache_storage.cache_conversation(conversation)

                # Also invalidate the user's conversations list cache to force a refresh next time
                cache_storage.invalidate_user_conversations_cache(conversation.user_id)

            return conversation_id

    async def get_user_conversations(self, user_id: str) -> List[Conversation]:
        # First try to get from cache
        cached_conversations = cache_storage.get_conversations_by_user_id_from_cache(
            user_id
        )
        if cached_conversations is not None:
            return cached_conversations

        # If not in cache, get from database
        async with self.session_factory() as session:
            result = await session.execute(
                text(
                    "SELECT id, user_id, title, created_at, updated_at"
                    " FROM conversations WHERE user_id = :user_id ORDER BY updated_at DESC"
                ),
                {"user_id": user_id},
            )
            rows = result.fetchall()
            return [Conversation(**dict(row._mapping)) for row in rows]

    async def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        # First try to get from cache
        cached_conversation = cache_storage.get_conversation_from_cache(conversation_id)
        if cached_conversation:
            return cached_conversation

        # If not in cache, get from database
        async with self.session_factory() as session:
            result = await session.execute(
                text(
                    "SELECT id, user_id, title, created_at, updated_at"
                    " FROM conversations WHERE id = :conversation_id"
                ),
                {"conversation_id": conversation_id},
            )
            row = result.one_or_none()
            if not row:
                return None

            conversation = Conversation(**dict(row._mapping))

            # Cache the result for future use
            try:
                cache_storage.cache_conversation(conversation)
            except Exception as e:
                logger.warning(f"Failed to cache conversation {conversation_id}: {e}")

            return conversation

    async def update_conversation_title(
        self,
        title: str,
        conversation_id: int,
        user_id: str,
    ) -> None:
        async with self.session_factory() as session:
            await session.execute(
                text(
                    "UPDATE conversations SET title = :title, updated_at = NOW() WHERE id = :conversation_id"
                ),
                {"title": title, "conversation_id": conversation_id},
            )
            await session.commit()

        # Update the cache - first get the cached conversation to update
        cached_conversation = cache_storage.get_conversation_from_cache(conversation_id)
        if cached_conversation:
            # Update the cached conversation and re-cache it
            cached_conversation.title = title
            cached_conversation.updated_at = datetime.now()
            cache_storage.cache_conversation(cached_conversation)
        else:
            # If not in cache, just invalidate cache to force refresh next time
            cache_storage.invalidate_conversation_cache(conversation_id)

        # Also invalidate the user's conversations list cache
        if user_id:
            cache_storage.invalidate_user_conversations_cache(user_id)

    async def delete_conversation(self, conversation_id: int) -> None:
        # Get user ID before deleting for cache invalidation
        conversation = cache_storage.get_conversation_from_cache(conversation_id)
        user_id = conversation.user_id if conversation else None

        async with self.session_factory() as session:
            await session.execute(
                text("DELETE FROM conversations WHERE id = :conversation_id"),
                {"conversation_id": conversation_id},
            )
            await session.commit()

        # Invalidate all related cache entries
        cache_storage.invalidate_conversation_cache(conversation_id)
        cache_storage.invalidate_conversation_messages_cache(conversation_id)
        cache_storage.invalidate_conversation_summaries_cache(conversation_id)

        # Also invalidate the user's conversations list cache
        if user_id:
            cache_storage.invalidate_user_conversations_cache(user_id)
