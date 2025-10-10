"""
Embedding Agent for generating semantic embeddings from text input.
Provides core business logic for embedding generation and vector operations.
"""

from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from db.userconfig_storage import UserConfigStorage

from models import ModelProfileType, PipelinePriority
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError


class EmbeddingAgent:
    """
    Embedding Agent for text-to-vector conversion with model profile integration.

    Provides core business logic for embedding generation using configured embedding models.
    Supports both single text and batch text embedding generation.
    """

    def __init__(self, pipeline_factory, user_config_storage: 'UserConfigStorage'):
        """
        Initialize embedding agent with dependency injection.

        Args:
            pipeline_factory: Factory for creating embedding pipelines
            user_config_storage: Injected UserConfigStorage service
        """
        self.pipeline_factory = pipeline_factory
        self.user_config_storage = user_config_storage
        self.logger = composer_logger.logger.bind(component="EmbeddingAgent")

    async def generate_embeddings(
        self, texts: List[str], user_id: str
    ) -> List[List[float]]:
        """
        Generate embeddings for input texts using configured embedding model.

        Args:
            texts: List of text strings to generate embeddings for
            user_id: User identifier for model profile retrieval

        Returns:
            List of embedding vectors (one per input text)
        """
        try:
            self.logger.info(
                "Generating embeddings",
                user_id=user_id,
                text_count=len(texts),
            )

            # Use injected storage service
            user_config_svc = self.user_config_storage
                
            # Lazy imports to avoid circular dependency
            from utils.model_profile import (  # pylint: disable=import-outside-toplevel
                get_model_profile_for_task,
            )
            from runner import (  # pylint: disable=import-outside-toplevel
                embed_pipeline,
                EmbeddingPipeline,
            )

            uc = await user_config_svc.get_user_config(user_id)
            # Get embedding model profile using standard pattern
            model_profile = await get_model_profile_for_task(
                uc.model_profiles, ModelProfileType.Embedding, user_id
            )
            circuit_breaker = model_profile.circuit_breaker or uc.circuit_breaker

            # Get embedding pipeline - embeddings need specialized pipeline
            pipeline = self.pipeline_factory.get_pipeline(
                model_profile,
                EmbeddingPipeline,
                PipelinePriority.NORMAL,
                circuit_breaker,
            )

            # Generate embeddings using specialized embed_pipeline function
            embeddings = await embed_pipeline(
                messages=texts,  # embed_pipeline accepts text directly
                pipeline=pipeline,
            )

            if embeddings:
                self.logger.info(
                    "Successfully generated embeddings",
                    user_id=user_id,
                    embedding_count=len(embeddings),
                    embedding_dimensions=len(embeddings[0]) if embeddings else 0,
                )
                return embeddings
            else:
                raise NodeExecutionError("No embeddings returned from pipeline")

        except Exception as e:
            self.logger.error(
                "Embedding generation failed",
                user_id=user_id,
                error=str(e),
                text_count=len(texts),
            )
            raise NodeExecutionError(f"Embedding generation failed: {e}") from e

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
