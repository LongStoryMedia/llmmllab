"""
Storage service for managing message content entities in the database.
Message contents represent the actual content parts of messages (text, URLs, etc.).
"""

from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="message_content_storage")


class MessageContentStorage:
    """Storage service for message content entities with CRUD operations."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self.session_factory = session_factory
        self.logger = logger

    async def add_content(
        self,
        content: MessageContent,
    ) -> Optional[int]:
        """
        Add a new message content to the database.

        Args:
            content: The MessageContent object to add

        Returns:
            The ID of the created message content, or None on failure
        """
        query = text(
            """
            INSERT INTO message_contents(message_id, type, text_content, url, format, name, created_at)
                VALUES (:message_id, :type, :text, :url, :format, :name, COALESCE(:created_at, NOW()))
            RETURNING id;
            """
        )

        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    query,
                    {
                        "message_id": content.message_id,
                        "type": content.type.value if hasattr(content.type, "value") else str(content.type),
                        "text": content.text,
                        "url": content.url,
                        "format": content.format,
                        "name": content.name,
                        "created_at": content.created_at,
                    },
                )
                row = result.mappings().first()

                if row:
                    content_id = row["id"]
                    self.logger.info(
                        f"Added message content {content_id} for message {content.message_id}"
                    )
                    await session.commit()
                    return content_id
                else:
                    self.logger.error(f"Failed to add content for message {content.message_id}")
                    return None

        except Exception as e:
            self.logger.error(
                f"Error adding content for message {content.message_id}: {e}"
            )
            return None

    async def get_contents_by_message(self, message_id: int) -> List[MessageContent]:
        """
        Retrieve all contents associated with a message.

        Args:
            message_id: ID of the message

        Returns:
            List of MessageContent objects
        """
        query = text(
            """
            SELECT
                mc.id,
                mc.message_id,
                mc.type,
                mc.text_content AS text,
                mc.url,
                mc.format,
                mc.name,
                mc.created_at
            FROM
                message_contents mc
            WHERE
                mc.message_id = :message_id
            ORDER BY
                mc.id
            """
        )

        try:
            async with self.session_factory() as session:
                result = await session.execute(query, {"message_id": message_id})
                rows = result.mappings().all()

                contents = []
                for row in rows:
                    try:
                        content = MessageContent(
                            id=row["id"],
                            message_id=row["message_id"],
                            type=MessageContentType(row["type"]),
                            text=row["text"],
                            url=row["url"],
                            format=row["format"],
                            name=row["name"],
                            created_at=row["created_at"],
                        )
                        contents.append(content)
                    except Exception as e:
                        self.logger.error(
                            f"Failed to parse content row {row['id']}: {e}"
                        )
                        continue

                self.logger.debug(
                    f"Retrieved {len(contents)} contents for message {message_id}"
                )
                return contents

        except Exception as e:
            self.logger.error(
                f"Error retrieving contents for message {message_id}: {e}"
            )
            return []

    async def delete_content(self, content_id: int) -> bool:
        """
        Delete a message content by ID.

        Args:
            content_id: ID of the content to delete

        Returns:
            True if successful, False otherwise
        """
        query = text(
            """
            DELETE FROM message_contents
            WHERE id = :id;
            """
        )

        try:
            async with self.session_factory() as session:
                result = await session.execute(query, {"id": content_id})
                await session.commit()

                if result.rowcount == 1:  # type: ignore[attr-defined]
                    self.logger.info(f"Deleted message content {content_id}")
                    return True
                else:
                    self.logger.warning(f"No content found with ID {content_id}")
                    return False

        except Exception as e:
            self.logger.error(f"Error deleting content {content_id}: {e}")
            return False

    async def delete_contents_by_message(self, message_id: int) -> bool:
        """
        Delete all contents associated with a message.

        Args:
            message_id: ID of the message

        Returns:
            True if successful, False otherwise
        """
        query = text(
            """
            DELETE FROM message_contents
            WHERE message_id = :message_id;
            """
        )

        try:
            async with self.session_factory() as session:
                result = await session.execute(query, {"message_id": message_id})
                await session.commit()

                self.logger.info(f"Deleted contents for message {message_id}: {result.rowcount} rows")  # type: ignore[attr-defined]
                return True

        except Exception as e:
            self.logger.error(f"Error deleting contents for message {message_id}: {e}")
            return False