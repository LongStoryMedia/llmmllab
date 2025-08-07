"""
Direct port of Maistro's conversation.go storage logic to Python with cache integration.
"""

from typing import List, Optional
from datetime import datetime
import asyncpg
from server.db.db_utils import typed_pool
import logging
from models.conversation import Conversation
from server.db.cache_storage import cache_storage
from server.db.db_utils import TypedPool, typed_pool

logger = logging.getLogger(__name__)


class ConversationStorage:
    def __init__(self, pool: asyncpg.Pool, get_query):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query

    async def create_conversation(self, user_id: str) -> Optional[int]:
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(
                self.get_query("conversation.create_conversation"), user_id, title
            )
            conversation_id = row["id"] if row and "id" in row else None

            # Cache the new conversation if successful
            if conversation_id:
                conversation = Conversation(
                    id=conversation_id,
                    user_id=user_id,
                    title=title,
                    model=model,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
                cache_storage.cache_conversation(conversation)

                # Also invalidate the user's conversations list cache to force a refresh next time
                cache_storage.invalidate_user_conversations_cache(user_id)

            return conversation_id

    async def get_user_conversations(self, user_id: str) -> List[Conversation]:
        # First try to get from cache
        cached_conversations = cache_storage.get_conversations_by_user_id_from_cache(
            user_id
        )
        if cached_conversations is not None:
            return cached_conversations

        # If not in cache, get from database
        async with self.typed_pool.acquire() as conn:
            rows = await conn.fetch(
                self.get_query("conversation.list_user_conversations"), user_id
            )
            return [Conversation(**dict(row)) for row in rows]

    async def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        # First try to get from cache
        cached_conversation = cache_storage.get_conversation_from_cache(conversation_id)
        if cached_conversation:
            return cached_conversation

        # If not in cache, get from database
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(
                self.get_query("conversation.get_conversation"), conversation_id
            )
            if not row:
                return None

            conversation = Conversation(**dict(row))

            # Cache the result for future use
            try:
                cache_storage.cache_conversation(conversation)
            except Exception as e:
                logger.warning(f"Failed to cache conversation {conversation_id}: {e}")

            return conversation

    async def update_conversation_title(self, conversation_id: int, title: str) -> None:
        async with self.typed_pool.acquire() as conn:
            await conn.execute(
                self.get_query("conversation.update_title"), title, conversation_id
            )

        # Update the cache - first get the conversation to update
        conversation = cache_storage.get_conversation_from_cache(conversation_id)
        if conversation:
            # Update the conversation and re-cache it
            conversation.title = title
            conversation.updated_at = datetime.now()
            cache_storage.cache_conversation(conversation)
        else:
            # If not in cache, invalidate to ensure next get will fetch from DB
            cache_storage.invalidate_conversation_cache(conversation_id)

        # Also invalidate the user's conversations list cache
        if conversation and conversation.user_id:
            cache_storage.invalidate_user_conversations_cache(conversation.user_id)

    async def delete_conversation(self, conversation_id: int) -> None:
        # Get user ID before deleting for cache invalidation
        conversation = cache_storage.get_conversation_from_cache(conversation_id)
        user_id = conversation.user_id if conversation else None

        async with self.typed_pool.acquire() as conn:
            await conn.execute(
                self.get_query("conversation.delete_conversation"), conversation_id
            )

        # Invalidate all related cache entries
        cache_storage.invalidate_conversation_cache(conversation_id)
        cache_storage.invalidate_conversation_messages_cache(conversation_id)
        cache_storage.invalidate_conversation_summaries_cache(conversation_id)

        # Also invalidate the user's conversations list cache
        if user_id:
            cache_storage.invalidate_user_conversations_cache(user_id)
