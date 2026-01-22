"""
module for managing search topic synthesis
"""

import json
import logging
from typing import Optional
import asyncpg
from db.db_utils import typed_pool
from models.search_topic_synthesis import SearchTopicSynthesis

logger = logging.getLogger(__name__)


class SearchStorage:
    """
    Class for managing search records in the database.
    """

    def __init__(self, pool: asyncpg.Pool, get_query):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query

    async def create(self, sts: SearchTopicSynthesis) -> Optional[int]:
        """
        Create a new search topic synthesis record.

        Args:
            sts: The SearchTopicSynthesis object to create

        Returns:
            The ID of the created synthesis, or None if creation failed
        """
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(
                self.get_query("search.add_search_topic_synthesis"),
                json.dumps(sts.urls),
                json.dumps(sts.topics),
                sts.synthesis,
                sts.conversation_id,
            )
            return row["id"] if row and "id" in row else None

    async def get_by_id(self, synthesis_id: int) -> Optional[SearchTopicSynthesis]:
        """
        Get a search topic synthesis record by its ID.

        Args:
            synthesis_id: The ID of the synthesis to retrieve

        Returns:
            The SearchTopicSynthesis object if found, or None if not found
        """
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(
                self.get_query("search.get_search_topic_synthesis"), synthesis_id
            )
            return (
                SearchTopicSynthesis(
                    id=row["id"],
                    urls=json.loads(row["urls"]),
                    topics=json.loads(row["topics"]),
                    synthesis=row["synthesis"],
                    created_at=row["created_at"],
                    conversation_id=row["conversation_id"],
                )
                if row
                else None
            )
