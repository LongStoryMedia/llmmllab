"""
Direct Nomic Embed Text v2 implementation using llama_cpp.Llama.
"""

import logging
import multiprocessing
import os
from typing import List, Optional

import llama_cpp
from langchain_core.embeddings import Embeddings

from models import Model, ModelProfile


class NomicEmbeddings(Embeddings):
    """
    Direct Nomic Embed Text v2 implementation using llama_cpp.Llama.

    Features:
    - Direct llama_cpp.Llama usage with embed() method
    - Task-specific prefixes (search_query: / search_document:)
    - 768-dimensional embeddings for ~100 languages
    - Optimized for 512 token context limit
    """

    def __init__(self, model: Model, profile: ModelProfile):
        """Initialize the Nomic embeddings with direct Llama instance."""
        self.model = model
        self.profile = profile
        self._logger = logging.getLogger(self.__class__.__name__)
        self.llama_instance: Optional[llama_cpp.Llama] = None
        self.embedding_dim = 768  # Nomic embedding dimension

    def _get_gguf_path(self) -> str:
        """Get the GGUF file path from model definition."""
        return (
            self.model.details.gguf_file
            if hasattr(self.model.details, "gguf_file") and self.model.details.gguf_file
            else self.model.model
        )

    def _get_optimal_threads(self) -> int:
        """Get optimal thread count based on system capabilities."""
        try:
            cpu_count = multiprocessing.cpu_count()
            optimal_threads = min(max(cpu_count // 2, 2), 8)
            return optimal_threads
        except Exception:
            return 4

    def _initialize_llama(self) -> llama_cpp.Llama:
        """Initialize the Llama instance for embedding generation."""
        if self.llama_instance is not None:
            return self.llama_instance

        gguf_path = self._get_gguf_path()

        # Check if model file exists
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"Embedding model file not found: {gguf_path}")

        try:
            # Use 512 context size for Nomic embeddings (model's maximum)
            context_size = min(self.profile.parameters.num_ctx or 512, 512)

            self.llama_instance = llama_cpp.Llama(
                model_path=gguf_path,
                n_ctx=context_size,
                n_gpu_layers=-1,  # Offload all layers to GPU
                n_threads=self._get_optimal_threads(),
                embedding=True,  # Enable embedding mode
                verbose=os.getenv("LOG_LEVEL", "WARNING").lower() == "debug",
                n_batch=512,
                seed=self.profile.parameters.seed or -1,
                use_mlock=False,
                f16_kv=True,
            )

            self._logger.info(f"Nomic embedding model initialized from: {gguf_path}")
            return self.llama_instance

        except Exception as e:
            self._logger.error(f"Failed to initialize Nomic embeddings: {e}")
            raise

    def _add_task_prefix(self, text: str, is_query: bool = False) -> str:
        """Add task-specific prefix for Nomic embeddings."""
        if is_query:
            return f"search_query: {text}"
        else:
            return f"search_document: {text}"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        if not texts:
            return []

        # Initialize model if needed
        if self.llama_instance is None:
            self.llama_instance = self._initialize_llama()

        embeddings = []
        for text in texts:
            # Add document prefix and generate embedding
            prefixed_text = self._add_task_prefix(text, is_query=False)
            embedding_result = self.llama_instance.embed(prefixed_text)

            # Extract embedding vector from response
            if isinstance(embedding_result, list) and len(embedding_result) > 0:
                if isinstance(embedding_result[0], list):
                    # Multiple embeddings returned, take the first
                    embeddings.extend(embedding_result)
                elif isinstance(embedding_result[0], (int, float)):
                    # Single embedding vector returned
                    embeddings.append(embedding_result)
                else:
                    # Fallback: zero vector
                    embeddings.append([0.0] * self.embedding_dim)
            else:
                # Fallback: zero vector
                embeddings.append([0.0] * self.embedding_dim)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        # Initialize model if needed
        if self.llama_instance is None:
            self.llama_instance = self._initialize_llama()

        # Add query prefix and generate embedding
        prefixed_text = self._add_task_prefix(text, is_query=True)
        embedding_result = self.llama_instance.embed(prefixed_text)

        # Extract embedding vector from response
        if isinstance(embedding_result, list) and len(embedding_result) > 0:
            if isinstance(embedding_result[0], list):
                # Multiple embeddings returned, take the first
                return embedding_result[0]
            elif isinstance(embedding_result[0], (int, float)):
                # Single embedding vector returned - cast to satisfy type checker
                return [
                    float(x) if isinstance(x, (int, float)) else 0.0
                    for x in embedding_result
                ]
            else:
                # Fallback: zero vector
                return [0.0] * self.embedding_dim
        else:
            # Fallback: zero vector
            return [0.0] * self.embedding_dim
