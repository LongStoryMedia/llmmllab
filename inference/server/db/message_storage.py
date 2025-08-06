"""
Direct port of Maistro's message.go storage logic to Python with cache integration.
"""

from calendar import c
from typing import List, Optional
import asyncpg
import logging
from datetime import datetime
from models.message import Message
from models.message_role import MessageRole
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from server.db.cache_storage import cache_storage
from server.db.db_utils import typed_pool

logger = logging.getLogger(__name__)


class MessageStorage:
    def __init__(self, pool: asyncpg.Pool, get_query):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query

    async def add_message(
        self, conversation_id: int, role: str, content: str
    ) -> Optional[int]:
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(
                self.get_query("message.add_message"), conversation_id, role, content
            )
            message_id = row["id"] if row and "id" in row else None

            # Cache the new message if successful
            if message_id:
                # Convert string role to MessageRole enum
                try:
                    message_role = MessageRole(role)
                except ValueError:
                    logger.warning(
                        f"Unknown message role: {role}, using system as default"
                    )
                    message_role = MessageRole.SYSTEM

                # Create MessageContent object with the text content
                message_content = [
                    MessageContent(type=MessageContentType.TEXT, text=content)
                ]

                message = Message(
                    id=message_id,
                    conversation_id=conversation_id,
                    role=message_role,
                    content=message_content,
                    created_at=datetime.now(),
                )
                cache_storage.cache_message(message)

                # Invalidate conversation messages list cache
                cache_storage.invalidate_conversation_messages_cache(conversation_id)

            return message_id

    async def get_message(self, message_id: int) -> Optional[Message]:
        # First try to get from cache
        cached_message = cache_storage.get_message_from_cache(message_id)
        if cached_message:
            return cached_message

        # If not in cache, get from database
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(self.get_query("message.get_message"), message_id)
            if not row:
                return None

            message = Message(**dict(row))

            # Cache the result for future use
            try:
                cache_storage.cache_message(message)
            except Exception as e:
                logger.warning(f"Failed to cache message {message_id}: {e}")

            return message

    async def get_conversation_history(self, conversation_id: int) -> List[Message]:
        # First try to get from cache
        cached_messages = cache_storage.get_messages_by_conversation_id_from_cache(
            conversation_id
        )
        if cached_messages is not None:
            return cached_messages

        # If not in cache, get from database
        async with self.typed_pool.acquire() as conn:
            rows = await conn.fetch(
                self.get_query("message.get_conversation_history"), conversation_id
            )
            messages = [Message(**dict(row)) for row in rows]

            # Cache the results for future use
            if messages:
                for msg in messages:
                    try:
                        c_rows = await conn.fetch(
                            self.get_query("message.get_content"), msg.id
                        )
                        msg.content = [
                            MessageContent(**dict(c_row)) for c_row in c_rows
                        ]
                    except Exception as e:
                        logger.warning(
                            f"Failed to create Message object for caching: {e}"
                        )

                cache_storage.cache_messages_by_conversation_id(
                    conversation_id, messages
                )

            return messages

    async def delete_message(self, message_id: int) -> None:
        # Get the message to find its conversation_id
        message = cache_storage.get_message_from_cache(message_id)
        conversation_id = message.conversation_id if message else None

        async with self.typed_pool.acquire() as conn:
            await conn.execute(self.get_query("message.delete_message"), message_id)

        # Invalidate message cache
        cache_storage.invalidate_message_cache(message_id)

        # Invalidate conversation messages list cache
        if conversation_id:
            cache_storage.invalidate_conversation_messages_cache(conversation_id)
