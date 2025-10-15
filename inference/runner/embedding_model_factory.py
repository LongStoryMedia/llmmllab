"""
Embedding model factory for creating Embeddings implementations.
"""

import logging
from typing import Optional

from langchain_core.embeddings import Embeddings

from models import Model, ModelProfile


class EmbeddingModelFactory:
    """
    Factory for creating Embeddings implementations.
    
    This factory returns Embeddings instances that can be used directly
    with LangChain's embedding utilities and vector stores.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("EmbeddingModelFactory initialized")

    def create_embedding_model(
        self,
        model: Model,
        profile: ModelProfile,
    ) -> Optional[Embeddings]:
        """Create an Embeddings implementation for the given model."""
        try:
            self.logger.info(f"Creating embedding model for {model.name} (task: {model.task})")

            # Only handle text-to-embeddings tasks
            if model.task != "TextToEmbeddings":
                self.logger.error(f"Unsupported task type for embedding model: {model.task}")
                return None

            return self._create_embedding_instance(model, profile)

        except Exception as e:
            self.logger.error(f"Error creating embedding model for {model.name}: {e}")
            return None

    def _create_embedding_instance(
        self,
        model: Model,
        profile: ModelProfile,
    ) -> Optional[Embeddings]:
        """Create an embedding model instance."""
        _ = profile  # Suppress unused argument warning
        
        if model.pipeline == "NomicEmbedTextPipe":
            try:
                # Create proper Embeddings wrapper for NomicEmbedTextPipe
                self.logger.warning("NomicEmbedTextPipe needs conversion to Embeddings interface")
                return None
                
            except Exception as e:
                self.logger.error(f"Failed to initialize NomicEmbedTextPipe: {e}")
                return None
                
        elif model.pipeline == "Qwen3EmbeddingPipe":
            try:
                # Create proper Embeddings wrapper for Qwen3EmbeddingPipe
                self.logger.warning("Qwen3EmbeddingPipe needs conversion to Embeddings interface")
                return None
                
            except Exception as e:
                self.logger.error(f"Failed to initialize Qwen3EmbeddingPipe: {e}")
                return None
                
        self.logger.warning(f"No embedding model implementation for pipeline: {model.pipeline}")
        return None


# Create global embedding model factory instance
embedding_model_factory = EmbeddingModelFactory()