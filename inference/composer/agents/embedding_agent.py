"""
Embedding Agent for generating semantic embeddings from text input.
Provides core business logic for embedding generation and vector operations.
"""

from typing import List

import numpy as np

from runner import PipelineFactory
from models import ModelProfile, PipelinePriority, NodeMetadata
from .base_agent import BaseAgent


class EmbeddingAgent(BaseAgent[List[List[float]]]):
    """
    Embedding Agent for text-to-vector conversion with model profile integration.

    Provides core business logic for embedding generation using configured embedding models.
    Supports both single text and batch text embedding generation.
    """

    def __init__(
        self,
        pipeline_factory: PipelineFactory,
        profile: ModelProfile,
        node_metadata: NodeMetadata,
    ):
        """
        Initialize embedding agent with required dependencies.

        Args:
            pipeline_factory: Factory for creating embedding pipelines
            profile: Model profile for embedding generation
            node_metadata: Node execution metadata for tracking
        """
        super().__init__(pipeline_factory, profile, node_metadata, "EmbeddingAgent")

    async def generate_embeddings(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Generate embeddings for input texts using configured embedding model.

        Args:
            texts: List of text strings to generate embeddings for
            user_id: User identifier for model profile retrieval

        Returns:
            List of embedding vectors (one per input text)
        """

        return await self.embed(
            messages=texts,
            priority=PipelinePriority.NORMAL,
        )

    async def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for single text input.

        Args:
            text: Text string to generate embedding for
            user_id: User identifier for model profile retrieval

        Returns:
            Single embedding vector
        """
        embeddings = await self.generate_embeddings([text])
        return embeddings[0] if embeddings else []

    async def compute_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embedding vectors.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            self.logger.error(f"Similarity computation failed: {e}")
            return 0.0
