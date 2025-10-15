"""
Embedding Model Factory for creating Embeddings instances.
Provides clean interface for retrieving embedding models from the pipeline factory.
"""

import logging
from typing import Optional

from langchain_core.embeddings import Embeddings
from models import Model, ModelProfile, CircuitBreakerConfig
from .pipeline_factory import pipeline_factory


class EmbeddingModelFactory:
    """Factory for creating Embeddings instances from the pipeline factory."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("EmbeddingModelFactory initialized")

    def create_embedding_model(
        self,
        model: Model,
        profile: ModelProfile,
        circuit_breaker: Optional[CircuitBreakerConfig] = None,
    ) -> Optional[Embeddings]:
        """
        Create an Embeddings instance from the pipeline factory.

        Args:
            model: Model configuration
            profile: Model profile for runtime settings
            circuit_breaker: Optional circuit breaker configuration

        Returns:
            Embeddings instance or None if creation fails
        """
        try:
            # Use the pipeline factory to create the embedding model
            pipeline = pipeline_factory.get_embedding_pipeline(profile)
            
            # Ensure it's an embeddings model
            if isinstance(pipeline, Embeddings):
                return pipeline
            else:
                self.logger.error(f"Pipeline for {model.name} is not an Embeddings model: {type(pipeline)}")
                return None

        except Exception as e:
            self.logger.error(f"Failed to create embedding model for {model.name}: {e}")
            return None


# Global embedding model factory instance
embedding_model_factory = EmbeddingModelFactory()