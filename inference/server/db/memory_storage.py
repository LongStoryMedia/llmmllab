"""
Direct port of Maistro's memory.go storage logic to Python.
"""

import math
import logging
from typing import List, Optional, Tuple
from datetime import datetime
import asyncpg
from server.db.db_utils import typed_pool
from models.memory import Memory, MemoryFragment
from models.message_role import MessageRole

logger = logging.getLogger(__name__)


class MemoryStorage:
    def __init__(self, pool: asyncpg.Pool, get_query):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query

    async def init_memory_schema(self):
        logger.info("Initializing memory schema...")
        async with self.typed_pool.acquire() as conn:
            await conn.execute(self.get_query("memory.init_memory_schema"))
            logger.info("Created memories table")
            await conn.execute(self.get_query("memory.create_memory_indexes"))
            logger.info("Created memory indexes")
            await conn.execute(self.get_query("memory.enable_memories_compression"))
            logger.info("Enabled memories compression")
            try:
                await conn.execute(self.get_query("memory.memories_compression_policy"))
                logger.info("Added memories compression policy")
            except Exception as e:
                logger.warning(f"Failed to add memories compression policy: {e}")
            try:
                await conn.execute(self.get_query("memory.memories_retention_policy"))
                logger.info("Added memories retention policy")
            except Exception as e:
                logger.warning(f"Failed to add memories retention policy: {e}")
            try:
                await conn.execute(
                    self.get_query("memory.create_memory_cascade_delete_triggers")
                )
                logger.info("Memory cascade delete trigger created successfully")
            except Exception as e:
                logger.warning(f"Failed to create memory cascade delete trigger: {e}")
        logger.info("Memory schema initialized successfully")

    async def store_memory(
        self,
        user_id: str,
        source: str,
        role: str,
        source_id: int,
        embeddings: List[List[float]],
    ):
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for embedding in embeddings:
                    pe, _ = self.process_embedding(embedding)
                    embedding_str = self.format_embedding_for_pgvector(pe)
                    await conn.execute(
                        self.get_query("memory.store_memory"),
                        user_id,
                        source_id,
                        source,
                        embedding_str,
                        role,
                    )

    async def delete_memory(self, id: str, user_id: str):
        async with self.typed_pool.acquire() as conn:
            await conn.execute(self.get_query("memory.delete_memory"), id, user_id)

    async def delete_all_user_memories(self, user_id: str):
        async with self.typed_pool.acquire() as conn:
            await conn.execute(
                self.get_query("memory.delete_all_user_memories"), user_id
            )

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
            # Prepare parameters for the SQL query
            params = [
                embedding_str,
                min_similarity,
                limit,
                user_id,
                conversation_id,
                start_date,
                end_date,
            ]
            async with self.typed_pool.acquire() as conn:
                rows = await conn.fetch(self.get_query("memory.search"), *params)
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
                            source=source_type,
                            created_at=created_at,
                            similarity=similarity,
                            source_id=source_id,
                            conversation_id=conversation_id_val,
                        )

                    if current_mem is not None:
                        current_mem.fragments.append(fragment)

                        if source_type == "summary":
                            memories.append(current_mem)
                            current_mem = None
                            last_pair_key = None
                        else:
                            last_pair_key = pair_key
                            if len(current_mem.fragments) == 2:
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
