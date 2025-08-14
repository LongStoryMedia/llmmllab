"""
Utility functions for working with embeddings.
"""

import json
import logging
import numpy as np
import os
from typing import List, Tuple, Dict, Any, Optional


class EmbeddingUtils:
    """Utility functions for embedding operations."""

    def __init__(self) -> None:
        """Initialize the embedding utilities."""
        self.logger = logging.getLogger(__name__)

    def cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            v1: First vector
            v2: Second vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        if not v1 or not v2:
            self.logger.warning("Empty vectors provided for cosine similarity")
            return 0.0

        try:
            # Convert lists to numpy arrays
            v1_arr = np.array(v1)
            v2_arr = np.array(v2)

            # Calculate cosine similarity
            dot_product = np.dot(v1_arr, v2_arr)
            norm_v1 = np.linalg.norm(v1_arr)
            norm_v2 = np.linalg.norm(v2_arr)

            if norm_v1 == 0 or norm_v2 == 0:
                self.logger.warning("Zero vector encountered in cosine similarity")
                return 0.0

            similarity = dot_product / (norm_v1 * norm_v2)
            return float(similarity)

        except Exception as e:
            self.logger.error(f"Error computing cosine similarity: {str(e)}")
            return 0.0
            self.logger.error(f"Error computing cosine similarity: {str(e)}")
            return 0.0

    def normalize_embedding(self, embedding: List[float]) -> List[float]:
        """
        Normalize an embedding vector to unit length.

        Args:
            embedding: Vector to normalize

        Returns:
            Normalized vector with unit length
        """
        try:
            # Convert to numpy array
            embedding_arr = np.array(embedding)

            # Calculate norm
            norm = np.linalg.norm(embedding_arr)

            # Normalize
            if norm > 0:
                normalized = embedding_arr / norm
                return normalized.tolist()
            else:
                self.logger.warning(
                    "Zero-length embedding encountered in normalization"
                )
                return embedding

        except Exception as e:
            self.logger.error(f"Error normalizing embedding: {str(e)}")
            return embedding

    def save_embeddings_to_jsonl(
        self, embeddings: List[List[float]], texts: List[str], output_path: str
    ) -> None:
        """
        Save embeddings and their source texts to a JSONL file.

        Args:
            embeddings: List of embeddings vectors
            texts: List of source texts
            output_path: Path to save the JSONL file
        """
        if len(embeddings) != len(texts):
            self.logger.error(
                f"Mismatch between embeddings ({len(embeddings)}) and texts ({len(texts)})"
            )
            return

        try:
            with open(output_path, "w") as f:
                for text, emb in zip(texts, embeddings):
                    record = {"text": text, "embedding": emb}
                    f.write(json.dumps(record) + "\n")

            self.logger.info(
                f"Successfully saved {len(embeddings)} embeddings to {output_path}"
            )

        except Exception as e:
            self.logger.error(f"Error saving embeddings to {output_path}: {str(e)}")

    def load_embeddings_from_jsonl(
        self, input_path: str
    ) -> Tuple[List[List[float]], List[str]]:
        """
        Load embeddings and their source texts from a JSONL file.

        Args:
            input_path: Path to the JSONL file

        Returns:
            Tuple of (embeddings, texts)
        """
        embeddings = []
        texts = []

        try:
            with open(input_path, "r") as f:
                for line in f:
                    record = json.loads(line)
                    if "text" in record and "embedding" in record:
                        texts.append(record["text"])
                        embeddings.append(record["embedding"])

            self.logger.info(
                f"Successfully loaded {len(embeddings)} embeddings from {input_path}"
            )
            return embeddings, texts

        except Exception as e:
            self.logger.error(f"Error loading embeddings from {input_path}: {str(e)}")
            return [], []

    def generate_embedding(
        self,
        text: str,
        model_path: str,
        is_query: bool = False,
        matryoshka_dim: Optional[int] = None,
    ) -> List[float]:
        """Generate an embedding for the given text.

        Args:
            text: Text to embed
            model_path: Path to the model file
            is_query: Whether this text is a query (True) or a document (False)
            matryoshka_dim: Optional dimension to truncate embeddings to

        Returns:
            Embedding vector as list of floats
        """
        # Use the adapter to generate the embedding
        try:
            from .adapter import EmbeddingModelAdapter

            adapter = EmbeddingModelAdapter()
            embeddings = adapter.generate_embeddings(
                texts=text,
                model_path=model_path,
                is_query=is_query,
                matryoshka_dim=matryoshka_dim,
            )

            if embeddings and len(embeddings) > 0:
                return embeddings[0]

            self.logger.error("Failed to generate embedding")
            return []
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            return []

    def calculate_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score between 0 and 1
        """
        # Use the existing cosine_similarity method
        return self.cosine_similarity(embedding1, embedding2)

    def generate_and_compare(
        self,
        query: str,
        documents: List[str],
        model_path: str,
        matryoshka_dim: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Generate embeddings for query and documents and calculate similarities.

        Args:
            query: Query text
            documents: List of document texts
            model_path: Path to the model file
            matryoshka_dim: Optional dimension to truncate embeddings to

        Returns:
            List of dictionaries with document text and similarity scores
        """
        # Generate query embedding
        query_embedding = self.generate_embedding(
            query, model_path, is_query=True, matryoshka_dim=matryoshka_dim
        )

        results = []
        for doc in documents:
            # Generate document embedding
            doc_embedding = self.generate_embedding(
                doc, model_path, is_query=False, matryoshka_dim=matryoshka_dim
            )

            # Calculate similarity
            similarity = self.calculate_similarity(query_embedding, doc_embedding)

            results.append({"document": doc, "similarity": similarity})

        # Sort by similarity (highest first)
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return results
