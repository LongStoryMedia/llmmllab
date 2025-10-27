"""
Storage service for managing thought entities in the database.
Thoughts represent AI assistant thinking/reasoning content linked to messages.
"""

import asyncpg
from typing import List, Optional
from datetime import datetime
from models.thought import Thought
from db.db_utils import TypedConnection, typed_pool
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="thought_storage")


class ThoughtStorage:
    """Storage service for thought entities with CRUD operations."""

    def __init__(self, pool: asyncpg.Pool, get_query):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query
        self.logger = llmmllogger.bind(component="thought_storage_instance")

    async def add_thought(
        self,
        message_id: int,
        text: str,
        created_at: Optional[datetime] = None,
        conn: Optional[TypedConnection] = None,
    ) -> Optional[int]:
        """
        Add a new thought to the database.

        Args:
            message_id: ID of the associated message
            text: The thinking/reasoning content
            created_at: Optional timestamp (defaults to NOW())

        Returns:
            The ID of the created thought, or None on failure
        """
        if created_at is None:
            created_at = datetime.utcnow()

        try:
            async with self.typed_pool.acquire() as conn:
                row = await conn.fetchrow(
                    self.get_query("thought.add_thought"), message_id, text, created_at
                )

                if row:
                    thought_id = row["id"]
                    self.logger.info(
                        f"Added thought {thought_id} for message {message_id}"
                    )
                    return thought_id
                else:
                    self.logger.error(f"Failed to add thought for message {message_id}")
                    return None

        except Exception as e:
            self.logger.error(f"Error adding thought for message {message_id}: {e}")
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
            async with self.typed_pool.acquire() as conn:
                rows = await conn.fetch(
                    self.get_query("thought.get_by_message"), message_id
                )

                thoughts = []
                for row in rows:
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
            async with self.typed_pool.acquire() as conn:
                result = await conn.execute(
                    self.get_query("thought.delete_by_message"), message_id
                )

                self.logger.info(f"Deleted thoughts for message {message_id}")
                return True

        except Exception as e:
            self.logger.error(f"Error deleting thoughts for message {message_id}: {e}")
            return False
