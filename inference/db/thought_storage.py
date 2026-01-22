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
        thought: Thought,
        conn: Optional[TypedConnection] = None,
    ) -> Optional[int]:
        """
        Add a new thought to the database.

        Args:
            thought: The Thought object to add to the database
            conn: Optional existing connection for transaction support

        Returns:
            The ID of the created thought, or None on failure
        """
        try:
            # Use provided connection or acquire a new one
            if conn is None:
                async with self.typed_pool.acquire() as connection:
                    return await self._add_thought(thought, connection)
            else:
                return await self._add_thought(thought, conn)

        except Exception as e:
            self.logger.error(
                f"Error adding thought for message {thought.message_id}: {e}"
            )
            return None

    async def _add_thought(
        self,
        thought: Thought,
        conn: TypedConnection,
    ) -> Optional[int]:
        """Internal method to add thought using a specific connection."""
        row = await conn.fetchrow(
            self.get_query("thought.add_thought"),
            thought.message_id,
            thought.text,
            thought.created_at,
        )

        if row:
            thought_id = row["id"]
            self.logger.info(
                f"Added thought {thought_id} for message {thought.message_id}"
            )
            return thought_id
        else:
            self.logger.error(f"Failed to add thought for message {thought.message_id}")
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
