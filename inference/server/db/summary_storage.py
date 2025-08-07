"""
Direct port of Maistro's summary.go storage logic to Python with cache integration.
"""

from typing import List, Optional
import asyncpg
from server.db.db_utils import typed_pool
import json
import logging
from datetime import datetime
from models.summary import Summary
from server.db.cache_storage import cache_storage

logger = logging.getLogger(__name__)


class SummaryStorage:
    def __init__(self, pool: asyncpg.Pool, get_query):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query

    async def create_summary(
        self, conversation_id: int, content: str, level: int, source_ids: List[int]
    ) -> Optional[int]:
        source_ids_json = json.dumps(source_ids)
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(
                self.get_query("summary.create_summary"),
                conversation_id,
                content,
                level,
                source_ids_json,
            )
            summary_id = row["id"] if row and "id" in row else None

            # Cache the new summary if successful
            if summary_id:
                summary = Summary(
                    id=summary_id,
                    conversation_id=conversation_id,
                    content=content,
                    level=level,
                    source_ids=source_ids,
                    created_at=datetime.now(),
                )
                cache_storage.cache_summary(summary)

                # Invalidate conversation summaries list cache
                cache_storage.invalidate_conversation_summaries_cache(conversation_id)

            return summary_id

    async def get_summaries_for_conversation(
        self, conversation_id: int
    ) -> List[Summary]:
        # First try to get from cache
        cached_summaries = cache_storage.get_summaries_by_conversation_id_from_cache(
            conversation_id
        )
        if cached_summaries is not None:
            return cached_summaries

        # If not in cache, get from database
        async with self.typed_pool.acquire() as conn:
            rows = await conn.fetch(
                self.get_query("summary.get_summaries_for_conversation"),
                conversation_id,
            )
            # Convert rows to Summary objects
            summaries = []
            for row in rows:
                # Convert row to dict and then to Summary
                row_dict = dict(row)
                # Parse source_ids from JSON if it's stored as a string
                if isinstance(row_dict.get("source_ids"), str):
                    row_dict["source_ids"] = json.loads(row_dict["source_ids"])
                summaries.append(Summary(**row_dict))

            # Cache the results for future use
            if summaries:
                summary_objects = []
                for sum in summaries:
                    summary_objects.append(sum)

                if summary_objects:
                    cache_storage.cache_summaries_by_conversation_id(
                        conversation_id, summary_objects
                    )

            return summaries

    async def get_recent_summaries(
        self, conversation_id: int, level: int, limit: int
    ) -> List[Summary]:
        # For this specialized query, we'll go directly to the database
        # since the cache might not have exactly what we need
        async with self.typed_pool.acquire() as conn:
            rows = await conn.fetch(
                self.get_query("summary.get_recent_summaries"),
                conversation_id,
                level,
                limit,
            )
            summaries = [Summary(**dict(row)) for row in rows]

            # We don't cache these specialized queries
            return summaries

    async def delete_summaries_for_conversation(self, conversation_id: int) -> None:
        async with self.typed_pool.acquire() as conn:
            await conn.execute(
                self.get_query("summary.delete_summaries"), conversation_id
            )

        # Invalidate conversation summaries cache
        cache_storage.invalidate_conversation_summaries_cache(conversation_id)

    async def get_summary(self, summary_id: int) -> Optional[Summary]:
        # First try to get from cache
        cached_summary = cache_storage.get_summary_from_cache(summary_id)
        if cached_summary:
            return cached_summary

        # If not in cache, get from database
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(self.get_query("summary.get_summary"), summary_id)
            if not row:
                return None

            summary = Summary(**dict(row))

            # Cache the result for future use
            try:
                cache_storage.cache_summary(summary)
            except Exception as e:
                logger.warning(f"Failed to cache summary {summary_id}: {e}")

            return summary
