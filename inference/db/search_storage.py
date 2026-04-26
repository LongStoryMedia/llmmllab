"""
module for managing search topic synthesis
"""

import json
import logging
from typing import Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from models.search_topic_synthesis import SearchTopicSynthesis

logger = logging.getLogger(__name__)


class SearchStorage:
    """
    Class for managing search records in the database.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self.session_factory = session_factory

    async def create(self, sts: SearchTopicSynthesis) -> Optional[int]:
        """
        Create a new search topic synthesis record.

        Args:
            sts: The SearchTopicSynthesis object to create

        Returns:
            The ID of the created synthesis, or None if creation failed
        """
        async with self.session_factory() as session:
            result = await session.execute(
                text("""
                    INSERT INTO search_topic_syntheses(urls, topics, synthesis, conversation_id, created_at)
                    VALUES (:urls, :topics, :synthesis, :conversation_id, NOW())
                    RETURNING id
                """),
                {
                    "urls": json.dumps(sts.urls),
                    "topics": json.dumps(sts.topics),
                    "synthesis": sts.synthesis,
                    "conversation_id": sts.conversation_id,
                },
            )
            await session.commit()
            row = result.mappings().first()
            return row["id"] if row and "id" in row else None

    async def get_by_id(self, synthesis_id: int) -> Optional[SearchTopicSynthesis]:
        """
        Get a search topic synthesis record by its ID.

        Args:
            synthesis_id: The ID of the synthesis to retrieve

        Returns:
            The SearchTopicSynthesis object if found, or None if not found
        """
        async with self.session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT id, urls, topics, synthesis, created_at
                    FROM search_topic_syntheses
                    WHERE id = :synthesis_id
                """),
                {"synthesis_id": synthesis_id},
            )
            row = result.mappings().first()
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