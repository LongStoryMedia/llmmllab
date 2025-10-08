"""
Simplified Nomic Embed Text v2 pipeline - pure embedding interface, no orchestration.
Replaced 777 lines of complex LangGraph orchestration with direct embedding calls.
"""

import os
import logging
from typing import List, Optional

from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

            gguf_path = self._get_gguf_path()

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
            self._logger.error(f"Failed to initialize Nomic embeddings: {e}")
            raise

    def _get_gguf_path(self) -> str:
        """Get the GGUF file path for Nomic Embed Text v2."""
        # Default root (matches .models.json entries). Can override via MODEL_PATH.
        base_path = os.getenv("MODEL_PATH", "/models")
        # Allow override for filename/relative path
        model_filename_env = os.getenv("NOMIC_EMBED_MODEL_FILENAME")
        if model_filename_env:
            model_path_candidate = model_filename_env
        else:
            # Default relative path under base_path
            model_path_candidate = "nomic-embed-text-v2-moe/nomic-embed-text-v2-moe.f16.gguf"

        if model_path_candidate.startswith("/"):
            gguf_path = model_path_candidate
        else:
            gguf_path = os.path.join(base_path, model_path_candidate)

        if not os.path.isfile(gguf_path):
            raise FileNotFoundError(
                "Nomic embedding model missing: {} (MODEL_PATH='{}', NOMIC_EMBED_MODEL_FILENAME='{}').".format(
                    gguf_path, base_path, model_filename_env or ""
                )
            )
        return gguf_path

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
