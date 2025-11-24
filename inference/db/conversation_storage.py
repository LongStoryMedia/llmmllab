"""
Direct port of Maistro's conversation.go storage logic to Python with cache integration.
"""

from typing import List, Optional
from datetime import datetime
import asyncpg
from models.conversation import Conversation
from db.cache_storage import cache_storage
from db.connection_recovery import ConnectionRecoveryManager, recovery_manager
from db.db_utils import typed_pool
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="conversation_storage")


class ConversationStorage:
    def __init__(self, pool: asyncpg.Pool, get_query, user_config_storage=None):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query
        self.user_config_storage = user_config_storage  # Will be set by Storage class
        
        # Initialize connection recovery manager
        self.recovery_manager = recovery_manager if recovery_manager else ConnectionRecoveryManager(pool)

    async def create_conversation(self, conversation: Conversation) -> Optional[int]:
        async with self.typed_pool.acquire() as conn:
            # Ensure the user exists with proper default config before creating the conversation
            if self.user_config_storage:
                await self.user_config_storage.ensure_user_exists(conversation.user_id)
            else:
                # Fallback to old method if UserConfigStorage not available with recovery
                async def _ensure_user():
                    return await conn.execute(
                        self.get_query("user.ensure_user"), conversation.user_id
                    )
                await self.recovery_manager.execute_with_recovery(_ensure_user)

            async def _create_conversation():
                return await conn.fetchrow(
                    self.get_query("conversation.create_conversation"),
                    conversation.user_id,
                    conversation.title,
                )
            
            row = await self.recovery_manager.execute_with_recovery(_create_conversation)
            assert row
            conversation_id = row["id"] if row and "id" in row else None

            # Cache the new conversation if successful
            if conversation_id:
                row_dict = dict(row)
                conversation = Conversation(**row_dict)
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
        async with self.typed_pool.acquire() as conn:
            async def _get_user_conversations():
                return await conn.fetch(
                    self.get_query("conversation.list_user_conversations"), user_id
                )
            
            rows = await self.recovery_manager.execute_with_recovery(_get_user_conversations)
            return [Conversation(**dict(row)) for row in rows]

    async def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        # First try to get from cache
        cached_conversation = cache_storage.get_conversation_from_cache(conversation_id)
        if cached_conversation:
            return cached_conversation

        # If not in cache, get from database
        async with self.typed_pool.acquire() as conn:
            async def _get_conversation():
                return await conn.fetchrow(
                    self.get_query("conversation.get_conversation"), conversation_id
                )
            
            row = await self.recovery_manager.execute_with_recovery(_get_conversation)
            if not row:
                return None

            conversation = Conversation(**dict(row))

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
        async with self.typed_pool.acquire() as conn:
            async def _update_title():
                return await conn.execute(
                    self.get_query("conversation.update_title"),
                    title,
                    conversation_id,
                )
            
            await self.recovery_manager.execute_with_recovery(_update_title)

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

        async with self.typed_pool.acquire() as conn:
            async def _delete_conversation():
                return await conn.execute(
                    self.get_query("conversation.delete_conversation"), conversation_id
                )
            
            await self.recovery_manager.execute_with_recovery(_delete_conversation)

        # Invalidate all related cache entries
        cache_storage.invalidate_conversation_cache(conversation_id)
        cache_storage.invalidate_conversation_messages_cache(conversation_id)
        cache_storage.invalidate_conversation_summaries_cache(conversation_id)

        # Also invalidate the user's conversations list cache
        if user_id:
            cache_storage.invalidate_user_conversations_cache(user_id)
