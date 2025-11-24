"""
Storage service for managing message content entities in the database.
Message contents represent the actual content parts of messages (text, URLs, etc.).
"""

import asyncpg
from typing import List, Optional
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from db.db_utils import TypedConnection, typed_pool
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="message_content_storage")


class MessageContentStorage:
    """Storage service for message content entities with CRUD operations."""

    def __init__(self, pool: asyncpg.Pool, get_query):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query
        self.logger = llmmllogger.bind(component="message_content_storage_instance")

    async def add_content(
        self,
        content: MessageContent,
        conn: Optional[TypedConnection] = None,
    ) -> Optional[int]:
        """
        Add a new message content to the database.

        Args:
            message_id: ID of the associated message
            content: The message content data
            created_at: Optional timestamp (defaults to NOW())
            conn: Optional existing connection for transaction support

        Returns:
            The ID of the created message content, or None on failure
        """
        try:
            # Use provided connection or acquire a new one
            if conn is None:
                async with self.typed_pool.acquire() as connection:
                    return await self._add_content(content, connection)
            else:
                return await self._add_content(content, conn)

        except Exception as e:
            self.logger.error(
                f"Error adding content for message {content.message_id}: {e}"
            )
            return None

    async def _add_content(
        self,
        content: MessageContent,
        conn: TypedConnection,
    ) -> Optional[int]:
        """Internal method to add message content using a specific connection."""
        row = await conn.fetchrow(
            self.get_query("message_content.add_content"),
            content.message_id,
            content.type.value if hasattr(content.type, "value") else str(content.type),
            content.text,
            content.url,
            content.created_at,
        )

        if row:
            content_id = row["id"]
            self.logger.info(
                f"Added message content {content_id} for message {content.message_id}"
            )
            return content_id
        else:
            self.logger.error(f"Failed to add content for message {content.message_id}")
            return None

    async def get_contents_by_message(self, message_id: int) -> List[MessageContent]:
        """
        Retrieve all contents associated with a message.

        Args:
            message_id: ID of the message

        Returns:
            List of MessageContent objects
        """
        try:
            async with self.typed_pool.acquire() as conn:
                rows = await conn.fetch(
                    self.get_query("message_content.get_by_message"), message_id
                )

                contents = []
                for row in rows:
                    row_dict = dict(row)
                    content = MessageContent(**row_dict)
                    contents.append(content)

                self.logger.debug(
                    f"Retrieved {len(contents)} contents for message {message_id}"
                )
                return contents

        except Exception as e:
            self.logger.error(
                f"Error retrieving contents for message {message_id}: {e}"
            )
            return []

    async def delete_contents_by_message(self, message_id: int) -> bool:
        """
        Delete all contents associated with a message.

        Args:
            message_id: ID of the message

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            async with self.typed_pool.acquire() as conn:
                await conn.execute(
                    self.get_query("message_content.delete_message_contents"),
                    message_id,
                )

                self.logger.info(f"Deleted contents for message {message_id}")
                return True

        except Exception as e:
            self.logger.error(f"Error deleting contents for message {message_id}: {e}")
            return False
