"""
Direct port of Maistro's summary.go storage logic to Python with cache integration.
"""

import json
import logging
from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from db.cache_storage import cache_storage
from models.summary import Summary

logger = logging.getLogger(__name__)


class SummaryStorage:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self.session_factory = session_factory

    async def create_summary(self, summary: Summary) -> Optional[int]:
        query = text(
            """
            INSERT INTO summaries(conversation_id, content, level, source_ids)
              VALUES (:conversation_id, :content, :level, :source_ids)
            RETURNING id;
            """
        )

        source_ids_json = json.dumps(summary.source_ids)

        async with self.session_factory() as session:
            result = await session.execute(
                query,
                {
                    "conversation_id": summary.conversation_id,
                    "content": summary.content,
                    "level": summary.level,
                    "source_ids": source_ids_json,
                },
            )
            row = result.mappings().first()
            summary_id = row["id"] if row and "id" in row else None

            if summary_id:
                await session.commit()

                # Cache the new summary if successful
                cache_storage.cache_summary(summary)

                # Invalidate conversation summaries list cache
                cache_storage.invalidate_conversation_summaries_cache(
                    summary.conversation_id
                )

            return summary_id

    async def get_summaries_for_conversation(
        self, conversation_id: int
    ) -> List[Summary]:
        """
        Retrieve all summaries for a given conversation, using cache if available.
        This method excludes summaries that have been consolidated into higher-level summaries.
        """
        # First try to get from cache
        cached_summaries = cache_storage.get_summaries_by_conversation_id_from_cache(
            conversation_id
        )
        if cached_summaries is not None:
            return cached_summaries

        query = text(
            """
            WITH referencing AS (
              SELECT DISTINCT
                (elem)::int AS ref_id
              FROM
                summaries s2
                CROSS JOIN LATERAL jsonb_array_elements_text(s2.source_ids) AS elem
              WHERE
                s2.conversation_id = :conversation_id
                AND s2.source_ids IS NOT NULL
            )
            SELECT
              s.id,
              s.conversation_id,
              s.content,
              s.level,
              s.source_ids,
              s.created_at
            FROM
              summaries s
            WHERE
              s.conversation_id = :conversation_id
              AND NOT EXISTS (
                SELECT
                  1
                FROM
                  referencing r
                WHERE
                  r.ref_id = s.id)
            ORDER BY
              s.level ASC,
              s.created_at DESC
            """
        )

        # If not in cache, get from database
        async with self.session_factory() as session:
            result = await session.execute(
                query, {"conversation_id": conversation_id}
            )
            rows = result.mappings().all()

            # Convert rows to Summary objects
            summaries = []
            for row in rows:
                row_dict = dict(row)
                # Parse source_ids from JSON if it's stored as a string
                if isinstance(row_dict.get("source_ids"), str):
                    row_dict["source_ids"] = json.loads(row_dict["source_ids"])
                summaries.append(Summary(**row_dict))

            # Cache the results for future use
            if summaries:
                cache_storage.cache_summaries_by_conversation_id(
                    conversation_id, summaries
                )

            return summaries

    async def get_recent_summaries(
        self, conversation_id: int, level: int, limit: int
    ) -> List[Summary]:
        # For this specialized query, we'll go directly to the database
        # since the cache might not have exactly what we need
        query = text(
            """
            SELECT
              id,
              conversation_id,
              content,
              level,
              source_ids,
              created_at
            FROM
              summaries
            WHERE
              conversation_id = :conversation_id
              AND level = :level
            ORDER BY
              created_at DESC
            LIMIT :limit
            """
        )

        async with self.session_factory() as session:
            result = await session.execute(
                query,
                {
                    "conversation_id": conversation_id,
                    "level": level,
                    "limit": limit,
                },
            )
            rows = result.mappings().all()
            summaries = [Summary(**dict(row)) for row in rows]

            # We don't cache these specialized queries
            return summaries

    async def delete_summaries_for_conversation(self, conversation_id: int) -> None:
        query = text(
            """
            DELETE FROM summaries
            WHERE conversation_id = :conversation_id;
            """
        )

        async with self.session_factory() as session:
            await session.execute(
                query, {"conversation_id": conversation_id}
            )
            await session.commit()

        # Invalidate conversation summaries cache
        cache_storage.invalidate_conversation_summaries_cache(conversation_id)

    async def get_summary(self, summary_id: int) -> Optional[Summary]:
        # First try to get from cache
        cached_summary = cache_storage.get_summary_from_cache(summary_id)
        if cached_summary:
            return cached_summary

        query = text(
            """
            SELECT
              id,
              conversation_id,
              content,
              level,
              source_ids,
              created_at
            FROM
              summaries
            WHERE
              id = :id
            """
        )

        # If not in cache, get from database
        async with self.session_factory() as session:
            result = await session.execute(query, {"id": summary_id})
            row = result.mappings().first()
            if not row:
                return None

            row_dict = dict(row)
            # Parse source_ids from JSON if it's stored as a string
            if isinstance(row_dict.get("source_ids"), str):
                row_dict["source_ids"] = json.loads(row_dict["source_ids"])
            summary = Summary(**row_dict)

            # Cache the result for future use
            try:
                cache_storage.cache_summary(summary)
            except Exception as e:
                logger.warning(f"Failed to cache summary {summary_id}: {e}")

            return summary