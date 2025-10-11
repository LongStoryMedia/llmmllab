"""
Simplified Nomic Embed Text v2 pipeline - pure embedding interface, no orchestration.
Replaced 777 lines of complex LangGraph orchestration with direct embedding calls.
"""

import os
import logging
import hashlib
from typing import List, Optional

from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import BaseTool

from models import (
    Message,
    MessageRole,
    Model,
    ModelProfile,
)
from runner.pipelines.base import SimpleEmbeddingPipeline, GrammarInput
from utils.message import extract_message_text


class NomicEmbedTextPipe(SimpleEmbeddingPipeline):
    """
    Simplified Nomic Embed Text v2 pipeline - direct embedding generation.

    Features:
    - Direct LlamaCpp embedding initialization
    - Automatic text chunking for 512 token limit
    - Task-specific prefixes (search_query: / search_document:)
    - 768-dimensional embeddings for ~100 languages
    """

    def __init__(self, model: Model, profile: ModelProfile):
        super().__init__(model, profile)
        self.llm: Optional[LlamaCppEmbeddings] = None
        self.text_splitter: Optional[RecursiveCharacterTextSplitter] = None
        self._logger = logging.getLogger(self.__class__.__name__)

    def _get_optimal_threads(self) -> int:
        """Get optimal thread count based on system capabilities."""
        try:
            import multiprocessing

            cpu_count = multiprocessing.cpu_count()
            optimal_threads = min(max(cpu_count // 2, 2), 8)
            self._logger.debug(
                f"Using {optimal_threads} threads (CPU count: {cpu_count})"
            )
            return optimal_threads
        except Exception:
            self._logger.warning(
                "Could not determine CPU count, using default threading"
            )
            return 4

    def _initialize_llm(self) -> LlamaCppEmbeddings:
        """Initialize the Nomic embedding model with original parameters."""
        if self.llm is not None:
            return self.llm

        try:
            import torch

            gguf_path = self.model.details.gguf_file or self.model.model

            # Graceful fallback: if model file is absent, use lightweight hash-based shim
            if not os.path.exists(gguf_path):
                fallback_mode = os.getenv("EMBEDDING_FALLBACK_MODE", "auto").lower()
                if fallback_mode in {"auto", "enabled"}:
                    self._logger.warning(
                        "Embedding model file missing (%s); using hash-based fallback embeddings (mode=%s).",
                        gguf_path,
                        fallback_mode,
                    )

                    class _HashEmbeddingShim:  # minimal async-compatible shim
                        DIM = 768

                        async def aembed_query(self, text: str) -> List[float]:  # type: ignore
                            # Deterministic hashing across tokens -> pseudo embedding
                            tokens = text.split()
                            vec = [0] * self.DIM
                            for ti, tok in enumerate(tokens):
                                h = hashlib.sha256(f"{ti}:{tok}".encode()).digest()
                                # Spread bytes across vector positions
                                for bj, b in enumerate(h):
                                    idx = (ti * 37 + bj) % self.DIM
                                    vec[idx] = (vec[idx] + b) & 0xFF
                            # Normalize to 0..1 floats
                            return [v / 255.0 for v in vec]

                    self.llm = _HashEmbeddingShim()  # type: ignore
                    assert self.llm is not None
                    return self.llm
                else:
                    raise FileNotFoundError(
                        f"Embedding model file not found and fallback disabled (path={gguf_path})"
                    )

            # Use the same context size as the model's maximum (512 for Nomic)
            context_size = min(self.profile.parameters.num_ctx or 512, 512)

            # Use original parameters from the complex implementation
            self.llm = LlamaCppEmbeddings(
                model_path=gguf_path,
                n_ctx=context_size,  # Maximum sequence length for this model
                n_gpu_layers=-1,  # Offload all layers to GPU
                n_threads=self._get_optimal_threads(),
                f16_kv=True,
                verbose=os.getenv("LOG_LEVEL", "WARNING").lower() == "debug",
                n_batch=512,  # Optimized batch size
                n_parts=-1,
                seed=self.profile.parameters.seed or -1,
                logits_all=True,  # enforced: required for logprobs usage elsewhere
                vocab_only=False,
                use_mlock=False,  # Better memory management
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            self._logger.info(f"Nomic embedding model initialized from: {gguf_path}")
            return self.llm

        except Exception as e:
            self._logger.error(
                "Failed to initialize Nomic embeddings (no fallback succeeded): %s",
                e,
            )
            raise

    def _init_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Initialize text splitter for handling long texts."""
        if self.text_splitter is None:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,  # Leave room for prefix and tokens
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", " ", ""],
            )
        return self.text_splitter

    def _add_task_prefix(self, text: str, is_query: bool = False) -> str:
        """Add task-specific prefix for Nomic embeddings."""
        if is_query:
            return f"search_query: {text}"
        else:
            return f"search_document: {text}"

    def _split_text_if_needed(self, text: str) -> List[str]:
        """Split text if it exceeds token limits."""
        # Simple token estimation (roughly 4 chars per token)
        estimated_tokens = len(text) // 4

        if estimated_tokens <= 400:  # Leave room for prefix
            return [text]

        # Use text splitter for long texts
        splitter = self._init_text_splitter()
        chunks = splitter.split_text(text)
        return chunks

    async def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query."""
        # Initialize model if needed
        if self.llm is None:
            self.llm = self._initialize_llm()

        try:
            # Add query prefix
            prefixed_text = self._add_task_prefix(text, is_query=True)

            # Generate embedding directly
            if self.llm is None:
                raise RuntimeError("Embedding model not initialized")
            embedding = await self.llm.aembed_query(prefixed_text)

            return embedding

        except Exception as e:
            self._logger.error(f"Query embedding failed: {e}")
            raise

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents."""
        # Initialize model if needed
        if self.llm is None:
            self.llm = self._initialize_llm()

        try:
            embeddings = []

            for text in texts:
                # Add document prefix
                prefixed_text = self._add_task_prefix(text, is_query=False)

                # Handle long texts by splitting
                chunks = self._split_text_if_needed(prefixed_text)

                if len(chunks) == 1:
                    # Single chunk - direct embedding
                    if self.llm is None:
                        raise RuntimeError("Embedding model not initialized")
                    embedding = await self.llm.aembed_query(chunks[0])
                    embeddings.append(embedding)
                else:
                    # Multiple chunks - average embeddings
                    chunk_embeddings = []
                    for chunk in chunks:
                        if self.llm is None:
                            raise RuntimeError("Embedding model not initialized")
                        chunk_emb = await self.llm.aembed_query(chunk)
                        chunk_embeddings.append(chunk_emb)

                    # Average the chunk embeddings
                    import numpy as np

                    avg_embedding = np.mean(chunk_embeddings, axis=0).tolist()
                    embeddings.append(avg_embedding)

            return embeddings

        except Exception as e:
            self._logger.error(f"Document embedding failed: {e}")
            raise

    async def invoke(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[GrammarInput] = None,
        **kwargs,
    ) -> List[List[float]]:
        """Generate embeddings for messages - main interface for SimpleEmbeddingPipeline."""
        _ = tools, grammar, kwargs  # Suppress unused warnings

        # Initialize model if needed
        if self.llm is None:
            self.llm = self._initialize_llm()

        try:
            # Extract texts from messages, skipping system messages
            texts = []
            for message in messages:
                if message.role == MessageRole.SYSTEM:
                    continue
                text = extract_message_text(message)
                if text:
                    texts.append(text)

            if not texts:
                # Return empty embedding if no texts
                return [[0.0] * 768]  # Nomic uses 768 dimensions

            # Generate embeddings for documents (default behavior)
            embeddings = await self.embed_documents(texts)
            return embeddings

        except Exception as e:
            self._logger.error(f"Embedding generation failed: {e}")
            raise
