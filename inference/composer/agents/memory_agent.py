"""
Memory Agent for semantic memory storage and retrieval.
Provides core business logic for memory operations and similarity search.
"""

from typing import List, Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from db.memory_storage import MemoryStorage

from models import Memory
from models.message_role import MessageRole
from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError
from utils.token_estimation import estimate_memory_tokens


class MemoryAgent:
    """
    Memory Agent for semantic memory operations.

    Provides business logic layer for memory operations, working with the storage layer
    for actual data persistence. Focuses on memory storage, retrieval, and formatting.
    Embedding operations are handled by separate embedding nodes in workflows.
    """

    def __init__(self, memory_storage: Optional['MemoryStorage'] = None):
        """Initialize memory agent with dependency injection.
        
        Args:
            memory_storage: Injected MemoryStorage service
        """
        self.memory_storage = memory_storage
        self.logger = composer_logger.logger.bind(component="MemoryAgent")

    async def store_memories(
        self,
        user_id: str,
        conversation_id: int,
        memories: List[Memory],
    ) -> bool:
        """
        Store messages as memories with their corresponding embeddings.

        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            messages: List of message data
            embeddings: Corresponding embeddings for messages

        Returns:
            True if storage succeeded, False otherwise
        """
        try:
            # Use injected storage or fallback to import
            if self.memory_storage:
                memory_svc = self.memory_storage
            else:
                from db import storage  # pylint: disable=import-outside-toplevel
                memory_svc = storage.get_service(storage.memory)

            # Store each message with its embedding
            success_count = 0
            for mem in memories:
                for frag in mem.fragments:
                    try:
                        if frag.embeddings is None:
                            self.logger.warning(
                                "Skipping memory fragment with missing embeddings",
                                user_id=user_id,
                                fragment_id=frag.id,
                            )
                            continue
                        await memory_svc.store_memory(
                            user_id=user_id,
                            source=mem.source,
                            role=frag.role,
                            source_id=mem.source_id,
                            embeddings=frag.embeddings,
                        )
                        success_count += 1
                    except Exception as e:
                        self.logger.warning(
                            "Failed to store individual memory fragment",
                            user_id=user_id,
                            fragment_id=frag.id,
                            error=str(e),
                        )

            self.logger.info(
                "Memory storage completed",
                user_id=user_id,
                conversation_id=conversation_id,
                total_messages=len(memories),
                successful_stores=success_count,
            )

            return success_count > 0

        except Exception as e:
            self.logger.error(
                "Memory storage failed",
                user_id=user_id,
                conversation_id=conversation_id,
                error=str(e),
            )
            return False

    async def search_memories_by_embedding(
        self,
        query_embeddings: List[List[float]],
        user_id: str,
        conversation_id: Optional[int] = None,
        max_results: int = 3,
        similarity_threshold: float = 0.7,
        enable_cross_conversation: bool = True,
        enable_cross_user: bool = False,
    ) -> List[Memory]:
        """
        Search conversation history using pre-computed embeddings.

        Uses storage layer for vector similarity search with provided embeddings.
        Embedding generation is handled by separate embedding nodes.

        Args:
            query_embeddings: Pre-computed query embeddings
            user_id: User identifier for filtering
            conversation_id: Optional conversation ID for filtering
            max_results: Maximum number of memories to return
            similarity_threshold: Minimum similarity score for inclusion
            enable_cross_conversation: Whether to search across conversations

        Returns:
            List of Memory objects with similarity scores and paired content
        """
        try:
            # Use injected storage or fallback to import
            if self.memory_storage:
                memory_svc = self.memory_storage
            else:
                from db import storage  # pylint: disable=import-outside-toplevel
                memory_svc = storage.get_service(storage.memory)

            self.logger.info(
                "Starting memory search with embeddings",
                user_id=user_id,
                conversation_id=conversation_id,
                embedding_count=len(query_embeddings),
                max_results=max_results,
                similarity_threshold=similarity_threshold,
                cross_conversation=enable_cross_conversation,
            )

            # Use storage layer for similarity search
            memories = await memory_svc.search_similarity(
                embeddings=query_embeddings,
                min_similarity=similarity_threshold,
                limit=max_results,
                user_id=(user_id if not enable_cross_user else None),
                conversation_id=(
                    conversation_id if not enable_cross_conversation else None
                ),
                start_date=None,
                end_date=None,
            )

            # Sort by relevance (storage layer already provides similarity scores)
            memories.sort(key=lambda m: m.similarity, reverse=True)
            final_memories = memories[:max_results]

            self.logger.info(
                "Memory search completed",
                user_id=user_id,
                results_count=len(memories),
                final_results=len(final_memories),
                avg_similarity=(
                    sum(m.similarity for m in final_memories) / len(final_memories)
                    if final_memories
                    else 0
                ),
            )

            return final_memories

        except Exception as e:
            self.logger.error(
                "Memory search failed",
                user_id=user_id,
                conversation_id=conversation_id,
                error=str(e),
            )
            raise NodeExecutionError(f"Memory search failed: {e}") from e
