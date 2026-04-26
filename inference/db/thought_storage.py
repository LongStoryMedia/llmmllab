"""
Storage service for managing thought entities in the database.
Thoughts represent AI assistant thinking/reasoning content linked to messages.
"""

from typing import List, Optional
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from models.thought import Thought
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="thought_storage")


class ThoughtStorage:
    """Storage service for thought entities with CRUD operations."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self.session_factory = session_factory
        self.logger = llmmllogger.bind(component="thought_storage_instance")

    async def add_thought(
        self,
        thought: Thought,
    ) -> Optional[int]:
        """
        Add a new thought to the database.

        Args:
            thought: The Thought object to add to the database

        Returns:
            The ID of the created thought, or None on failure
        """
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    text("""
                        INSERT INTO thoughts(message_id, text, created_at)
                        VALUES (:message_id, :text, COALESCE(:created_at, NOW()))
                        RETURNING id, message_id, text, created_at
                    """),
                    {
                        "message_id": thought.message_id,
                        "text": thought.text,
                        "created_at": thought.created_at,
                    },
                )
                await session.commit()

                row = result.mappings().first()
                if row:
                    thought_id = row["id"]
                    self.logger.info(
                        f"Added thought {thought_id} for message {thought.message_id}"
                    )
                    return thought_id
                else:
                    self.logger.error(f"Failed to add thought for message {thought.message_id}")
                    return None

        except Exception as e:
            self.logger.error(
                f"Error adding thought for message {thought.message_id}: {e}"
            )
            return None

    async def get_thoughts_by_message(self, message_id: int) -> List[Thought]:
        """
        Retrieve all thoughts associated with a message.

        Args:
            message_id: ID of the message

        Returns:
            List of Thought objects
        """
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    text("""
                        SELECT
                            id,
                            message_id,
                            text,
                            created_at
                        FROM
                            thoughts
                        WHERE
                            message_id = :message_id
                        ORDER BY
                            created_at ASC
                    """),
                    {"message_id": message_id},
                )

                thoughts = []
                for row in result.mappings():
                    thought = Thought(
                        id=row["id"],
                        message_id=row["message_id"],
                        text=row["text"],
                        created_at=row["created_at"],
                    )
                    thoughts.append(thought)

                self.logger.debug(
                    f"Retrieved {len(thoughts)} thoughts for message {message_id}"
                )
                return thoughts

        except Exception as e:
            self.logger.error(
                f"Error retrieving thoughts for message {message_id}: {e}"
            )
            return []

    async def delete_thoughts_by_message(self, message_id: int) -> bool:
        """
        Delete all thoughts associated with a message.

        Args:
            message_id: ID of the message

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    text("DELETE FROM thoughts WHERE message_id = :message_id"),
                    {"message_id": message_id},
                )
                await session.commit()

                self.logger.info(f"Deleted thoughts for message {message_id}")
                return result.rowcount > 0

        except Exception as e:
            self.logger.error(f"Error deleting thoughts for message {message_id}: {e}")
            return False