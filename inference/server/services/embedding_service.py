"""
Embedding service for generating and managing vector embeddings.
"""

from typing import List, Optional
import os
import asyncio
import numpy as np
from pydantic import SecretStr

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from ..config import logger


class EmbeddingService:
    """
    Service for generating embeddings for text.
    """

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the embedding model to use.
            api_key: API key for the embedding service (if needed).
        """
        self.model_name = model_name or os.environ.get(
            "EMBEDDING_MODEL", "text-embedding-ada-002"
        )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        # Initialize the embedding model
        self._initialize_embedding_model()

    def _initialize_embedding_model(self) -> None:
        """Initialize the embedding model based on the provided configuration."""
        try:
            if "ada" in self.model_name.lower() or "openai" in self.model_name.lower():
                # Use OpenAI embeddings - convert api_key to SecretStr if it exists
                api_key_secret = SecretStr(self.api_key) if self.api_key else None

                self.embeddings = OpenAIEmbeddings(
                    model=self.model_name, api_key=api_key_secret
                )
                logger.info(f"Initialized OpenAI embedding model: {self.model_name}")
            else:
                # Use HuggingFace embeddings
                self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
                logger.info(
                    f"Initialized HuggingFace embedding model: {self.model_name}"
                )
        except (ValueError, ImportError) as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            # Fall back to a default model that should be available
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info(
                "Initialized fallback embedding model: sentence-transformers/all-MiniLM-L6-v2"
            )

    async def create_embeddings(self, text: str) -> List[List[float]]:
        """
        Create embeddings for the given text.

        Args:
            text: The text to embed

        Returns:
            A list of embedding vectors (typically just one)
        """
        if not text:
            logger.warning("Empty text provided for embedding")
            # Return a zero vector with the appropriate dimensionality
            return [[0.0] * 1536]  # 1536 is typical for OpenAI embeddings

        try:
            # Execute in a separate thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self.embeddings.embed_documents([text])
            )
            logger.debug(f"Created embedding with dimension {len(result[0])}")
            return result
        except (ValueError, RuntimeError, ImportError) as e:
            logger.error(f"Error creating embedding: {str(e)}")
            # Return a zero vector with the appropriate dimensionality
            return [[0.0] * 1536]

    async def create_query_embedding(self, text: str) -> List[float]:
        """
        Create a query embedding for the given text.
        This might use a different model than document embeddings in some cases.

        Args:
            text: The query text to embed

        Returns:
            An embedding vector
        """
        if not text:
            logger.warning("Empty text provided for query embedding")
            return [0.0] * 1536  # 1536 is typical for OpenAI embeddings

        try:
            # Execute in a separate thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self.embeddings.embed_query(text)
            )
            return result
        except (ValueError, RuntimeError, ImportError) as e:
            logger.error(f"Error creating query embedding: {str(e)}")
            return [0.0] * 1536

    def compute_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """
        Compute the cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        # Convert to numpy arrays for efficient computation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
