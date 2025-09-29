"""
Direct port of Maistro's message.go storage logic to Python with cache integration.
"""

from typing import List, Optional, Union
import logging
from datetime import datetime
import asyncpg
from models.message import Message
from models.message_role import MessageRole
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from db.cache_storage import cache_storage
from db.db_utils import TypedConnection, typed_pool

logger = logging.getLogger(__name__)


class MessageStorage:
    def __init__(self, pool: asyncpg.Pool, get_query):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query
        self.logger = logging.getLogger(__name__)

    async def add_message(self, message: Message) -> Optional[int]:
        # Process content to ensure it's in the right format for storage
        assert message.conversation_id, "Message must have a conversation_id"

        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(
                self.get_query("message.add_message"),
                message.conversation_id,
                message.role,
            )
            message_id = row["id"] if row and "id" in row else None

            for c in message.content:
                await conn.execute(
                    self.get_query("message.add_content"),
                    message_id,
                    c.type,
                    c.text,
                    c.url,
                )

            cache_storage.cache_message(message)
            # Invalidate conversation messages list cache
            cache_storage.invalidate_conversation_messages_cache(
                message.conversation_id
            )

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
        cached_messages = cache_storage.get_conversation_messages(conversation_id)
        if cached_messages:
            # Validate cached messages before returning
            validated_messages = []
            # Ensure cached_messages is iterable
            if not isinstance(cached_messages, list):
                cached_messages = [cached_messages]

            for msg in cached_messages:
                # Ensure content is a list
                if not msg.content:
                    msg.content = [
                        MessageContent(type=MessageContentType.TEXT, text="")
                    ]
                elif not isinstance(msg.content, list):
                    msg.content = [
                        MessageContent(
                            type=MessageContentType.TEXT, text=str(msg.content)
                        )
                    ]

                validated_messages.append(msg)

            return validated_messages

        # If not in cache, get from database
        async with self.typed_pool.acquire() as conn:
            rows = await conn.fetch(
                self.get_query("message.get_conversation_history"),
                conversation_id,
            )
            message_dicts = [dict(row) for row in rows]

            messages: List[Message] = (
                await self._build_messages(conversation_id, message_dicts, conn)
                if message_dicts
                else []
            )
            if messages:
                cache_storage.cache_conversation_messages(conversation_id, messages)

            return messages

    async def get_messages_by_conversation_id(
        self, conversation_id: int, limit: int, offset: int
    ) -> List[Message]:
        # Check cache first
        cached_messages = cache_storage.get_messages_by_conversation_id_from_cache(
            conversation_id
        )
        if cached_messages:
            return cached_messages

        # If not in cache, get from database
        async with self.typed_pool.acquire() as conn:
            rows = await conn.fetch(
                self.get_query("message.get_by_conversation_id"),
                conversation_id,
                limit,
                offset,
            )
            message_dicts = [dict(row) for row in rows]
            messages: List[Message] = (
                await self._build_messages(conversation_id, message_dicts, conn)
                if message_dicts
                else []
            )
            if messages:
                cache_storage.cache_messages_by_conversation_id(
                    conversation_id, messages
                )

            return messages

    async def delete_message(self, message_id: int) -> None:
        # Get the message to find its conversation_id
        message = await self.get_message(message_id)
        if not message:
            logger.warning(f"Message {message_id} not found and could not be deleted")
            return

            # Invalidate message cache
            cache_storage.invalidate_message_cache(message_id)

        # Invalidate conversation messages list cache
        if message.conversation_id:
            cache_storage.invalidate_conversation_messages_cache(
                message.conversation_id
            )

    async def _build_messages(
        self, conversation_id: int, message_dicts: List[dict], conn: TypedConnection
    ) -> List[Message]:
        """Build a Message object from a database row."""
        messages: List[Message] = []
        for msg in message_dicts:
            try:
                c_rows = await conn.fetch(
                    self.get_query("message.get_content"), msg["id"]
                )
                msg_content_dicts = [dict(c_row) for c_row in c_rows]

                # Ensure content is a list of MessageContent objects
                if not msg_content_dicts:
                    # If no content rows, create a default MessageContent with empty text
                    msg["content"] = [
                        MessageContent(type=MessageContentType.TEXT, text="", url=None)
                    ]
                else:
                    msg["content"] = [
                        MessageContent(
                            type=d.get("type", MessageContentType.TEXT),
                            text=d.get("text_content", ""),
                            url=d.get("url"),
                        )
                        for d in msg_content_dicts
                    ]

                # Ensure conversation_id is set
                if "conversation_id" not in msg or msg["conversation_id"] is None:
                    msg["conversation_id"] = conversation_id

                # Create the Message object
                message_obj = Message(**msg)
                messages.append(message_obj)
            except Exception as e:
                logger.warning(
                    f"Failed to create Message object for caching: {e}, msg={msg}"
                )

        return messages
