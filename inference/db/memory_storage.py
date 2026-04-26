"""
Direct port of Maistro's memory.go storage logic to Python.
"""

import math
import logging
from typing import List, Optional, Tuple
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from models.memory import Memory, MemoryFragment
from models.message_role import MessageRole
from models.memory_source import MemorySource

logger = logging.getLogger(__name__)

# SQL queries loaded from db/sql/memory/
_SEARCH_SQL = """WITH
similar_messages_unfiltered AS (
    SELECT
        m.id AS source_id,
        m.conversation_id,
        m.role,
        m.created_at,
        1 -(e.embedding <=> :embedding_vec) AS similarity
    FROM
        memories e
        JOIN messages m ON e.source_id = m.id
    WHERE
        e.source = 'message'
        AND 1 -(e.embedding <=> :embedding_vec) > :min_sim
        AND (:user_id::text IS NULL
            OR e.user_id = :user_id::text)
            AND (:start_date::text IS NULL
                OR m.created_at >=(:start_date::text)::timestamptz)
            AND (:end_date::text IS NULL
                OR m.created_at <=(:end_date::text)::timestamptz)
),
similar_summaries_unfiltered AS (
    SELECT
        s.id AS source_id,
        s.conversation_id,
        1 -(e.embedding <=> :embedding_vec) AS similarity
    FROM
        memories e
        JOIN summaries s ON e.source_id = s.id
    WHERE
        e.source = 'summary'
        AND 1 -(e.embedding <=> :embedding_vec) > :min_sim
        AND (:user_id::text IS NULL
            OR e.user_id = :user_id::text)
            AND (:start_date::text IS NULL
                OR s.created_at >=(:start_date::text)::timestamptz)
            AND (:end_date::text IS NULL
                OR s.created_at <=(:end_date::text)::timestamptz)
),
similar_search_topics_unfiltered AS (
    SELECT
        st.id AS source_id,
        st.conversation_id,
        1 -(e.embedding <=> :embedding_vec) AS similarity
    FROM
        memories e
        JOIN search_topic_syntheses st ON e.source_id = st.id
    WHERE
        e.source = 'search'
        AND 1 -(e.embedding <=> :embedding_vec) > :min_sim
        AND (:user_id::text IS NULL
            OR e.user_id = :user_id::text)
            AND (:start_date::text IS NULL
                OR st.created_at >=(:start_date::text)::timestamptz)
            AND (:end_date::text IS NULL
                OR st.created_at <=(:end_date::text)::timestamptz)
),
similar_documents_unfiltered AS (
    SELECT
        d.id AS source_id,
        m.conversation_id,
        1 -(e.embedding <=> :embedding_vec) AS similarity
    FROM
        memories e
        JOIN documents d ON e.source_id = d.id
        JOIN messages m ON d.message_id = m.id
    WHERE
        e.source = 'document'
        AND 1 -(e.embedding <=> :embedding_vec) > :min_sim
        AND (:user_id::text IS NULL
            OR e.user_id = :user_id::text)
            AND (:start_date::text IS NULL
                OR d.created_at >=(:start_date::text)::timestamptz)
            AND (:end_date::text IS NULL
                OR d.created_at <=(:end_date::text)::timestamptz)
),
similar_messages AS (
    SELECT
        *
    FROM
        similar_messages_unfiltered
    WHERE
(:conversation_id::integer IS NULL)
        OR
(conversation_id = :conversation_id::integer)
),
message_pairs AS (
    SELECT
        user_msg.source_id AS first_message_id,
        'user' AS first_message_role,
        user_msg.created_at AS first_message_created_at,
(
            SELECT
                m.id
            FROM
                messages m
            WHERE
                m.conversation_id = user_msg.conversation_id
                AND m.role = 'assistant'
                AND m.created_at > user_msg.created_at
            ORDER BY
                m.created_at ASC
            LIMIT 1) AS second_message_id,
        'assistant' AS second_message_role,
(
            SELECT
                m.created_at
            FROM
                messages m
            WHERE
                m.conversation_id = user_msg.conversation_id
                AND m.role = 'assistant'
                AND m.created_at > user_msg.created_at
            ORDER BY
                m.created_at ASC
            LIMIT 1) AS second_message_created_at,
        user_msg.conversation_id,
        user_msg.similarity,
        'user_matched' AS pair_type
    FROM
        similar_messages user_msg
    WHERE
        user_msg.role = 'user'
    UNION ALL
    SELECT
        (
            SELECT
                m.id
            FROM
                messages m
            WHERE
                m.conversation_id = assistant_msg.conversation_id
                AND m.role = 'user'
                AND m.created_at < assistant_msg.created_at
            ORDER BY
                m.created_at DESC
            LIMIT 1) AS first_message_id,
        'user' AS first_message_role,
(
            SELECT
                m.created_at
            FROM
                messages m
            WHERE
                m.conversation_id = assistant_msg.conversation_id
                AND m.role = 'user'
                AND m.created_at < assistant_msg.created_at
            ORDER BY
                m.created_at DESC
            LIMIT 1) AS first_message_created_at,
        assistant_msg.source_id AS second_message_id,
        'assistant' AS second_message_role,
        assistant_msg.created_at AS second_message_created_at,
        assistant_msg.conversation_id,
        assistant_msg.similarity,
        'assistant_matched' AS pair_type
    FROM
        similar_messages assistant_msg
    WHERE
        assistant_msg.role = 'assistant'
),
deduplicated_message_pairs AS (
    SELECT
        first_message_id,
        first_message_role,
        first_message_created_at,
        second_message_id,
        second_message_role,
        second_message_created_at,
        conversation_id,
        similarity,
        pair_type,
        ROW_NUMBER() OVER (PARTITION BY first_message_id,
            second_message_id ORDER BY similarity DESC) AS exact_pair_rank,
        ROW_NUMBER() OVER (PARTITION BY first_message_id ORDER BY similarity DESC) AS first_message_rank,
        ROW_NUMBER() OVER (PARTITION BY second_message_id ORDER BY similarity DESC) AS second_message_rank
    FROM
        message_pairs
),
message_results_to_fetch AS (
    SELECT
        first_message_id AS source_id,
        'message' AS source_type,
        similarity,
        conversation_id,
        1 AS pair_order, -- User message first
        similarity AS original_similarity,
        CONCAT(COALESCE(first_message_id::text, 'null'), '-', COALESCE(second_message_id::text, 'null')) AS pair_key -- Create unique pair key
    FROM
        deduplicated_message_pairs
    WHERE
        exact_pair_rank = 1 -- Only the highest similarity for this exact pair
        AND first_message_rank = 1 -- Only include this message in one pair (highest similarity)
        AND first_message_id IS NOT NULL -- Must have a first message
        AND (
            second_message_id IS NOT NULL
            AND second_message_rank = 1
            AND first_message_role = 'user'
            AND second_message_role = 'assistant')
    UNION ALL
    SELECT
        second_message_id AS source_id,
        'message' AS source_type,
        similarity, -- Use same similarity as original
        conversation_id,
        2 AS pair_order, -- Assistant message second
        similarity AS original_similarity,
        CONCAT(COALESCE(first_message_id::text, 'null'), '-', COALESCE(second_message_id::text, 'null')) AS pair_key -- Same pair key to match
    FROM
        deduplicated_message_pairs
    WHERE
        exact_pair_rank = 1 -- Only the highest similarity for this exact pair
        AND first_message_rank = 1 -- Only include this message in one pair (highest similarity)
        AND second_message_rank = 1 -- Only include this message in one pair (highest similarity)
        AND first_message_id IS NOT NULL -- Must have a first message
        AND second_message_id IS NOT NULL -- Must have a second message for pairs
        AND first_message_role = 'user' -- Verify it's a user message
        AND second_message_role = 'assistant' -- Verify it's paired with an assistant message
),
summary_results_to_fetch AS (
    SELECT
        ssu.source_id,
        'summary' AS source_type,
        ssu.similarity,
        ssu.conversation_id,
        0 AS pair_order, -- Summaries are standalone
        ssu.similarity AS original_similarity,
        CONCAT('summary-', ssu.source_id) AS pair_key -- Each summary is its own group
    FROM
        similar_summaries_unfiltered ssu
    WHERE
(:conversation_id::integer IS NULL)
    OR (ssu.conversation_id = :conversation_id::integer)
),
search_results_to_fetch AS (
    SELECT
        ss.source_id,
        'search' AS source_type,
        ss.similarity,
        ss.conversation_id,
        0 AS pair_order, -- Search results are standalone
        ss.similarity AS original_similarity,
        CONCAT('search-', ss.source_id) AS pair_key -- Each search result is its own group
    FROM
        similar_search_topics_unfiltered ss
    WHERE
(:conversation_id::integer IS NULL)
    OR (ss.conversation_id = :conversation_id::integer)
),
document_results_to_fetch AS (
    SELECT
        sa.source_id,
        'document' AS source_type,
        sa.similarity,
        sa.conversation_id,
        0 AS pair_order, -- documents are standalone
        sa.similarity AS original_similarity,
        CONCAT('document-', sa.source_id) AS pair_key -- Each document is its own group
    FROM
        similar_documents_unfiltered sa
    WHERE
(:conversation_id::integer IS NULL)
    OR (sa.conversation_id = :conversation_id::integer)
),
all_results_to_fetch AS (
    SELECT
        *
    FROM
        message_results_to_fetch
    UNION ALL
    SELECT
        *
    FROM
        summary_results_to_fetch
    UNION ALL
    SELECT
        *
    FROM
        search_results_to_fetch
    UNION ALL
    SELECT
        *
    FROM
        document_results_to_fetch
),
unique_results AS (
    SELECT
        source_id,
        source_type,
        similarity,
        conversation_id,
        pair_order,
        pair_key,
        original_similarity,
        ROW_NUMBER() OVER (ORDER BY original_similarity DESC,
            pair_key,
            pair_order) AS global_rank
FROM
    all_results_to_fetch
),
limited_pairs AS (
    SELECT
        pair_key
    FROM
        unique_results
    GROUP BY
        pair_key
    ORDER BY
        MAX(similarity) DESC
    LIMIT :limit
)
SELECT
    COALESCE(m.role, 'system') AS role,
    u.source_id,
    COALESCE(mc.text_content, s.content, ss.synthesis, d.text_content, d.filename) AS content,
    u.source_type,
    u.similarity,
    COALESCE(m.conversation_id, s.conversation_id, ss.conversation_id, msg.conversation_id) AS conversation_id,
    COALESCE(m.created_at, s.created_at, ss.created_at, d.created_at) AS created_at
FROM
    unique_results u
    LEFT JOIN messages m ON u.source_id = m.id
        AND u.source_type = 'message'
    LEFT JOIN message_contents mc ON m.id = mc.message_id
        AND u.source_type = 'message'
    LEFT JOIN summaries s ON u.source_id = s.id
        AND u.source_type = 'summary'
    LEFT JOIN search_topic_syntheses ss ON u.source_id = ss.id
        AND u.source_type = 'search'
    LEFT JOIN documents d ON u.source_id = d.id
        AND u.source_type = 'document'
    LEFT JOIN messages msg ON d.message_id = msg.id
        AND u.source_type = 'document'
WHERE
    u.pair_key IN (
        SELECT
            pair_key
        FROM
            limited_pairs)
ORDER BY
    u.similarity DESC, -- Sort by highest similarity first
    COALESCE(m.conversation_id, s.conversation_id, ss.conversation_id, msg.conversation_id), -- Keep conversation pairs together
    COALESCE(m.created_at, s.created_at, ss.created_at, d.created_at) -- Maintain chronological order within conversations
"""
_TRIGGERS_SQL = """-- Trigger function to delete memories when a message is deleted
CREATE OR REPLACE FUNCTION delete_memories_on_message_delete()
  RETURNS TRIGGER
  AS $$
BEGIN
  DELETE FROM memories
  WHERE source = 'message'
    AND source_id = OLD.id;
  RETURN OLD;
END;
$$
LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS cascade_delete_memories_on_message ON messages;

CREATE TRIGGER cascade_delete_memories_on_message
  BEFORE DELETE ON messages
  FOR EACH ROW
  EXECUTE FUNCTION delete_memories_on_message_delete();

-- Trigger function to delete memories when a summary is deleted
CREATE OR REPLACE FUNCTION delete_memories_on_summary_delete()
  RETURNS TRIGGER
  AS $$
BEGIN
  DELETE FROM memories
  WHERE source = 'summary'
    AND source_id = OLD.id;
  RETURN OLD;
END;
$$
LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS cascade_delete_memories_on_summary ON summaries;

CREATE TRIGGER cascade_delete_memories_on_summary
  BEFORE DELETE ON summaries
  FOR EACH ROW
  EXECUTE FUNCTION delete_memories_on_summary_delete();

-- Trigger function to delete memories when a search_topic_synthesis is deleted
CREATE OR REPLACE FUNCTION delete_memories_on_search_topic_synthesis_delete()
  RETURNS TRIGGER
  AS $$
BEGIN
  DELETE FROM memories
  WHERE source = 'search'
    AND source_id = OLD.id;
  RETURN OLD;
END;
$$
LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS cascade_delete_memories_on_search_topic_synthesis ON search_topic_syntheses;

CREATE TRIGGER cascade_delete_memories_on_search_topic_synthesis
  BEFORE DELETE ON search_topic_syntheses
  FOR EACH ROW
  EXECUTE FUNCTION delete_memories_on_search_topic_synthesis_delete();"""
_COMPRESSION_SQL = """-- Enable compression on memories hypertable
ALTER TABLE memories SET (timescaledb.compress, timescaledb.compress_segmentby = 'user_id, source');"""
_COMP_POLICY_SQL = """-- Add data compression policy for memories
SELECT
  add_compression_policy('memories', INTERVAL '30 days', if_not_exists => TRUE);"""
_RETENTION_SQL = """-- Add retention policy for memories data (365 days)
SELECT
  add_retention_policy('memories', INTERVAL '365 days', if_not_exists => TRUE);"""
_INDEXES_SQL = """-- Create memory indexes for efficient search
CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_id_createdat_unique ON memories(id, created_at);

CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);

CREATE INDEX idx_memories_source_id_source ON memories(source_id, source);

-- Create vector similarity search index on memories
CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING HNSW(embedding vector_cosine_ops);

SET max_parallel_workers_per_gather = 4;"""



class MemoryStorage:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self.session_factory = session_factory

    async def init_memory_schema(self):
        logger.info("Initializing memory schema...")
        async with self.session_factory() as session:
            # First create the base schema
            await session.execute(text(
                "CREATE TABLE IF NOT EXISTS memories("
                "id serial, user_id text NOT NULL, source_id integer NOT NULL,"
                " source text NOT NULL, role text, embedding vector(768) NOT NULL,"
                " created_at timestamptz NOT NULL DEFAULT NOW(),"
                " FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,"
                " PRIMARY KEY (id, created_at))"
            ))
            await session.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_memories_source_id ON memories(source_id)"
            ))
            await session.execute(text(
                "SELECT create_hypertable('memories', 'created_at',"
                " if_not_exists => TRUE, migrate_data => TRUE,"
                " chunk_time_interval => interval '30 days')"
            ))
            await session.commit()
            logger.info("Created memories table")

            # Set up triggers before compression
            try:
                await session.execute(text(_TRIGGERS_SQL))
                logger.info("Memory cascade delete trigger created successfully")
            except Exception as e:
                logger.warning(f"Failed to create memory cascade delete trigger: {e}")

            # Enable compression before setting policies
            try:
                await session.execute(text(_COMPRESSION_SQL))
                logger.info("Enabled memories compression")
            except Exception as e:
                logger.warning(f"Failed to enable memories compression: {e}")

            # Set compression and retention policies
            try:
                await session.execute(text(_COMP_POLICY_SQL))
                logger.info("Added memories compression policy")
            except Exception as e:
                logger.warning(f"Failed to add memories compression policy: {e}")

            try:
                await session.execute(text(_RETENTION_SQL))
                logger.info("Added memories retention policy")
            except Exception as e:
                logger.warning(f"Failed to add memories retention policy: {e}")

            # Create indexes last
            try:
                await session.execute(text(_INDEXES_SQL))
                logger.info("Created memory indexes")
            except Exception as e:
                logger.warning(f"Failed to create memory indexes: {e}")

            await session.commit()

        logger.info("Memory schema initialized successfully")

    async def store_memory(
        self,
        user_id: str,
        source: str,
        role: str,
        source_id: int,
        embeddings: List[List[float]],
    ):
        async with self.session_factory() as session:
            for embedding in embeddings:
                pe, _ = self.process_embedding(embedding)
                embedding_str = self.format_embedding_for_pgvector(pe)
                await session.execute(
                    text(
                        "INSERT INTO memories(user_id, source_id, source, embedding, role)"
                        " VALUES (:user_id, :source_id, :source, :embedding, :role)"
                    ),
                    {
                        "user_id": user_id,
                        "source_id": source_id,
                        "source": source,
                        "embedding": embedding_str,
                        "role": role,
                    },
                )
            await session.commit()

    async def delete_memory(self, id: str, user_id: str):
        async with self.session_factory() as session:
            await session.execute(
                text("DELETE FROM memories WHERE id = :mem_id AND user_id = :user_id"),
                {"mem_id": id, "user_id": user_id},
            )
            await session.commit()

    async def delete_all_user_memories(self, user_id: str):
        async with self.session_factory() as session:
            await session.execute(
                text("DELETE FROM memories WHERE user_id = :user_id"),
                {"user_id": user_id},
            )
            await session.commit()

    async def search_similarity(
        self,
        embeddings: List[List[float]],
        min_similarity: float,
        limit: int,
        user_id: Optional[str] = None,
        conversation_id: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Memory]:
        """
        Search for semantically similar messages and summaries, grouping fragments into Memory objects.
        """

        memories = []
        if not embeddings:
            return memories

        for embedding in embeddings:
            if not embedding:
                continue
            embedding_str = self.format_embedding_for_pgvector(embedding)
            # Prepare parameters for the SQL query (now includes conversation_id)
            async with self.session_factory() as session:
                result = await session.execute(
                    text(_SEARCH_SQL),
                    {
                        "embedding_vec": embedding_str,
                        "min_sim": min_similarity,
                        "limit": limit,
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "start_date": start_date,
                        "end_date": end_date,
                    },
                )
                rows = result.mappings().all()
                current_mem = None
                last_pair_key = None
                for row in rows:
                    role = row["role"]
                    source_id = row["source_id"]
                    content = row["content"]
                    source_type = row["source_type"]
                    similarity = float(row["similarity"])
                    conversation_id_val = row["conversation_id"]
                    created_at = row["created_at"]

                    # Determine pair key for grouping
                    if source_type == "summary":
                        pair_key = f"summary-{source_id}"
                    else:
                        if role == "user":
                            pair_key = f"pair-{source_id}"
                        else:
                            pair_key = last_pair_key

                    # Ensure role is of type MessageRole

                    fragment = MemoryFragment(
                        id=source_id, role=MessageRole(role), content=content
                    )

                    if pair_key != last_pair_key or source_type == "summary":
                        if current_mem and current_mem.fragments:
                            memories.append(current_mem)
                        current_mem = Memory(
                            fragments=[],
                            source=MemorySource.SUMMARY,
                            created_at=created_at,
                            similarity=similarity,
                            source_id=source_id,
                            conversation_id=conversation_id_val,
                        )

                    if current_mem is not None:
                        current_mem.fragments = list(current_mem.fragments) + [fragment]

                        if source_type == "message":
                            last_pair_key = pair_key
                            if len(current_mem.fragments) == 2:
                                memories.append(current_mem)
                                current_mem = None
                                last_pair_key = None
                        else:
                            memories.append(current_mem)
                            current_mem = None
                            last_pair_key = None

                # Add any remaining memory
                if current_mem is not None and current_mem.fragments:
                    memories.append(current_mem)

        return memories

    @staticmethod
    def format_embedding_for_pgvector(embedding: List[float]) -> str:
        return "[" + ",".join(f"{val:f}" for val in embedding) + "]"

    @staticmethod
    def process_embedding(embedding: List[float]) -> Tuple[List[float], int]:
        original_dimension = len(embedding)
        target_dimension = 768
        if original_dimension == target_dimension:
            return embedding, original_dimension
        elif original_dimension < target_dimension:
            return (
                MemoryStorage.pad_vector(embedding, target_dimension),
                original_dimension,
            )
        else:
            return (
                MemoryStorage.reduce_vector(embedding, target_dimension),
                original_dimension,
            )

    @staticmethod
    def pad_vector(vec: List[float], target_dimension: int) -> List[float]:
        return vec + [0.0] * (target_dimension - len(vec))

    @staticmethod
    def reduce_vector(vec: List[float], target_dimension: int) -> List[float]:
        original_dimension = len(vec)
        result = [0.0] * target_dimension
        ratio = original_dimension / target_dimension
        for i in range(target_dimension):
            start_idx = int(math.floor(i * ratio))
            end_idx = min(int(math.floor((i + 1) * ratio)), original_dimension)
            if start_idx >= end_idx:
                if i < original_dimension:
                    result[i] = vec[i]
                continue
            result[i] = sum(vec[start_idx:end_idx]) / (end_idx - start_idx)
        return MemoryStorage.normalize_vector(result)

    @staticmethod
    def normalize_vector(vec: List[float]) -> List[float]:
        s = sum(v * v for v in vec)
        if s < 1e-10:
            return vec
        magnitude = math.sqrt(s)
        return [v / magnitude for v in vec]
