"""
Embedding Agent for generating semantic embeddings from text input.
Provides core business logic for embedding generation and vector operations.
"""

from typing import List, cast

import numpy as np

from runner import PipelineFactory
from models import ModelProfile, PipelinePriority, NodeMetadata
from composer.core.errors import NodeExecutionError
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

    async def execute_pipeline(
        self, stream: bool = False, **kwargs
    ) -> List[List[float]]:
        """
        Execute embedding generation pipeline with the provided parameters.

        This is the standard interface for pipeline execution required by BaseAgent.

        Args:
            stream: Whether to stream the response (not applicable for embeddings)
            **kwargs: Pipeline execution parameters, expected to include:
                - texts: List of text strings to generate embeddings for
                - user_id: User identifier

        Returns:
            List[List[float]]: The generated embeddings
        """
        texts = kwargs.get("texts", [])
        user_id = kwargs.get("user_id", "")

        if not texts:
            raise NodeExecutionError(
                "texts parameter is required for embedding generation"
            )

        return await self.generate_embeddings(texts=texts, user_id=user_id)

    async def generate_embeddings(
        self,
        texts: List[str],
        user_id: str,
    ) -> List[List[float]]:
        """
        Generate embeddings for input texts using configured embedding model.

        Args:
            texts: List of text strings to generate embeddings for
            user_id: User identifier for model profile retrieval

        Returns:
            List of embedding vectors (one per input text)
        """
        async def _execute_embedding_pipeline(user_id: str, text_count: int) -> List[List[float]]:
            """Internal pipeline executor for embeddings."""
            # Use injected storage service
            from runner import (  # pylint: disable=import-outside-toplevel
                embed_pipeline,
                EmbeddingPipeline,
            )

            # Get embedding pipeline - embeddings need specialized pipeline
            pipeline = self.pipeline_factory.get_pipeline(
                self.profile,
                List[List[float]],
                PipelinePriority.NORMAL,
            )

            # Generate embeddings using specialized embed_pipeline function
            embeddings = await embed_pipeline(
                messages=texts,  # embed_pipeline accepts text directly
                pipeline=cast(EmbeddingPipeline, pipeline),
            )

            if embeddings:
                return embeddings
            else:
                raise NodeExecutionError("No embeddings returned from pipeline")

        # Use BaseAgent's generic pipeline runner with metadata
        return await self.run_generic_pipeline_with_metadata(
            pipeline_executor=_execute_embedding_pipeline,
            operation_name="embedding_generation",
            user_id=user_id,
            text_count=len(texts),
        )

    async def generate_single_embedding(self, text: str, user_id: str) -> List[float]:
        """
        Generate embedding for single text input.

        Args:
            text: Text string to generate embedding for
            user_id: User identifier for model profile retrieval

        Returns:
            Single embedding vector
        """
        embeddings = await self.generate_embeddings([text], user_id)
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

    def validate_embeddings(self, embeddings: List[List[float]]) -> bool:
        """
        Validate that embeddings are properly formatted.

        Args:
            embeddings: List of embedding vectors to validate

        Returns:
            True if valid, False otherwise
        """
        if not embeddings or not isinstance(embeddings, list):
            return False

        if not all(
            isinstance(emb, list) and all(isinstance(x, (int, float)) for x in emb)
            for emb in embeddings
        ):
            return False

        # Check that all embeddings have the same dimension
        if len(set(len(emb) for emb in embeddings)) > 1:
            return False

        return True
