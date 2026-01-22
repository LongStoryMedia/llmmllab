"""
Embedding extractor for working with Nomic Embed Text model.

This file provides classes for extracting embeddings from text using Nomic Embed Text model.
"""

import logging
from typing import List, Dict, Any, Optional

from .adapter import EmbeddingModelAdapter
from .utils import EmbeddingUtils


class EmbeddingExtractor:
    """Class for extracting embeddings from text using Nomic Embed models."""

    def __init__(self) -> None:
        """Initialize the embedding extractor."""
        self.logger = logging.getLogger(__name__)
        self.utils = EmbeddingUtils()
        self.adapter = EmbeddingModelAdapter()

    def extract_embedding(
        self,
        text: str,
        model_path: str,
        is_query: bool = False,
        matryoshka_dim: Optional[int] = None,
    ) -> List[float]:
        """
        Extract embedding for a single text.

        Args:
            text: Text to extract embedding from
            model_path: Path to the model file
            is_query: Whether the text is a query (True) or document (False)
            matryoshka_dim: Optional dimension for Matryoshka embedding truncation

        Returns:
            Embedding vector as list of floats
        """
        self.logger.info(f"Extracting embedding for text (length={len(text)})")

        try:
            # Use the adapter to get the embedding
            embeddings = self.adapter.generate_embeddings(
                texts=text,
                model_path=model_path,
                is_query=is_query,
                matryoshka_dim=matryoshka_dim,
            )

            if embeddings and len(embeddings) > 0:
                return embeddings[0]

            # If we failed to get an embedding, return empty vector
            self.logger.error("Failed to extract embedding")
            return []

        except Exception as e:
            self.logger.error(f"Error extracting embedding: {str(e)}")
            return []

    def extract_embeddings(
        self,
        texts: List[str],
        model_path: str,
        is_query: bool = False,
        matryoshka_dim: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Extract embeddings for multiple texts.

        Args:
            texts: List of texts to extract embeddings from
            model_path: Path to the model file
            is_query: Whether the texts are queries (True) or documents (False)
            matryoshka_dim: Optional dimension for Matryoshka embedding truncation

        Returns:
            List of embedding vectors
        """
        self.logger.info(f"Extracting embeddings for {len(texts)} texts")

        try:
            # Use the adapter to get embeddings
            embeddings = self.adapter.generate_embeddings(
                texts=texts,
                model_path=model_path,
                is_query=is_query,
                matryoshka_dim=matryoshka_dim,
            )

            return embeddings

        except Exception as e:
            self.logger.error(f"Error extracting embeddings: {str(e)}")
            # Return empty embeddings as fallback
            return [[] for _ in texts]
