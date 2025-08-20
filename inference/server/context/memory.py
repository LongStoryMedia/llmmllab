"""
Memory retrieval functionality for RAG system.
"""

from typing import List, Optional
from datetime import datetime

from runner.pipelines.base_pipeline import Embeddings
from models import Memory, UserConfig
from server.db import storage
from server.config import logger


class MemoryContext:
    """
    Context for retrieving relevant memories based on query embeddings.
    """

    retrieved_memories: List[Memory]

    def __init__(self, user_cfg: UserConfig, conversation_id: int):
        """
        Initialize the memory context.

        Args:
            user_cfg: The user configuration
            conversation_id: The ID of the conversation
        """
        self.user_config = user_cfg
        self.user_id = user_cfg.user_id
        self.conversation_id = conversation_id
        self.logger = logger
        self.retrieved_memories = []

    async def retrieve_memories(
        self,
        embeddings: Embeddings,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Memory]:
        """
        Retrieve relevant memories for the provided embeddings.

        Args:
            embeddings: The embeddings to search with
            start_date: Optional start date filter for memories
            end_date: Optional end date filter for memories

        Returns:
            List of retrieved memories
        """
        if self.retrieved_memories:
            return self.retrieved_memories

        try:
            # Search for memories with the embeddings
            memories = await storage.get_service(storage.memory).search_similarity(
                embeddings,
                min_similarity=self.user_config.memory.similarity_threshold,
                limit=self.user_config.memory.limit,
                user_id=(
                    self.user_id
                    if not self.user_config.memory.enable_cross_user
                    else None
                ),
                conversation_id=(
                    self.conversation_id
                    if not self.user_config.memory.enable_cross_conversation
                    else None
                ),
                start_date=start_date,
                end_date=end_date,
            )
            self.retrieved_memories = memories
            return memories
        except Exception as e:
            self.logger.error(f"Error retrieving memories: {e}")
            return []
