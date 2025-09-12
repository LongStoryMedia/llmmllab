"""
Clean Qwen3-Embedding-0.6B pipeline with essential functionality.
"""

import logging
import os
import re
from typing import List, Optional, Tuple
import torch
import numpy as np
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from models import (
    Model,
    Message,
    ModelProfile,
)
from ..llamacpp.base_llamacpp import BaseLlamaCppCore
from langchain_core.tools import BaseTool
from utils.message import extract_message_text


logger = logging.getLogger(__name__)


class Qwen3EmbeddingPipe(BaseLlamaCppCore):
    """Clean pipeline for Qwen3-Embedding-0.6B model with optimized implementation."""

    llm: LlamaCppEmbeddings

    def __init__(self, model: Model, profile: ModelProfile):
        """Initialize the Qwen3 Embedding pipeline."""
        # Initialize with list as the expected return type for embeddings
        super().__init__(
            model,
            profile,
            expected_return_type=list,
            model_size_category="medium",
        )

        self.logger = logging.getLogger(__name__)

        # Qwen3-specific parameters
        self.max_context_tokens = 8192
        self.embedding_dim = 1024

        # Validate model definition
        if not (model.details and model.model):
            raise ValueError(
                "Model definition for Qwen3EmbeddingPipe must include model path details."
            )

        # Get and validate GGUF file
        gguf_path = self._get_gguf_path()
        self._validate_gguf_file(gguf_path)

        # Initialize model with optimizations
        self._initialize_model(gguf_path)

        # Initialize text splitter for handling long texts
        self._init_text_splitter()

    def _init_text_splitter(self) -> None:
        """Initialize the text splitter with conservative token estimates."""
        # Use conservative character-to-token ratio (3:1) to avoid exceeding limits
        max_chunk_chars = self.max_context_tokens * 3

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_chars,
            chunk_overlap=max_chunk_chars // 10,  # 10% overlap
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            keep_separator=True,
        )

    async def process_messages(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        is_tool_generation: bool = False,
    ) -> List[List[float]]:
        """Process messages and return embeddings (no tools/graph needed)."""
        # Embedding pipelines don't use tools or tool generation flags
        _ = tools, is_tool_generation  # Acknowledge unused parameters

        texts: List[str] = []
        for message in messages:
            text = extract_message_text(message)
            if text:
                texts.append(text)
        if not texts:
            return [[0.0] * self.embedding_dim]
        return await self._generate_embeddings_with_splitting(
            texts, aggregation_method="mean"
        )

    def create_graph(self, tools: Optional[List[BaseTool]] = None):  # type: ignore[override]
        """Embeddings pipeline doesn't use LangGraph workflows."""
        raise NotImplementedError("Embedding pipelines do not use LangGraph graphs")

    def _get_gguf_path(self) -> str:
        """Get the GGUF file path from model definition."""
        return (
            self.model.details.gguf_file
            if hasattr(self.model.details, "gguf_file") and self.model.details.gguf_file
            else self.model.model
        )

    def _validate_gguf_file(self, gguf_path: str) -> None:
        """Validate the GGUF file exists and has reasonable size."""
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

        file_size = os.path.getsize(gguf_path)
        if file_size < 1_000_000:  # Less than 1MB is suspicious
            raise ValueError(
                f"GGUF file is too small ({file_size} bytes), likely a placeholder: {gguf_path}"
            )

    def _initialize_model(self, gguf_path: str) -> None:
        """Initialize the LangChain LlamaCppEmbeddings model with optimizations."""
        try:
            # Get optimal settings from profile
            context_size = min(self.profile.parameters.num_ctx or 8192, 8192)

            self.llm = LlamaCppEmbeddings(
                model_path=gguf_path,
                n_ctx=context_size,
                n_gpu_layers=-1,  # Offload all layers to GPU
                n_threads=self._get_optimal_threads(),
                f16_kv=True,
                verbose=os.getenv("LOG_LEVEL", "WARNING").lower() == "debug",
                n_batch=512,
                n_parts=-1,
                seed=self.profile.parameters.seed or -1,
                logits_all=True,  # enforced
                vocab_only=False,
                use_mlock=False,  # Better memory management
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        except Exception as e:
            self.logger.error(f"Error initializing {self.__class__.__name__}: {str(e)}")
            raise

    def _get_optimal_threads(self) -> int:
        """Get optimal thread count based on system capabilities."""
        try:
            import multiprocessing

            cpu_count = multiprocessing.cpu_count()
            # Use half the CPU cores, capped at 8 for optimal performance
            optimal_threads = min(max(cpu_count // 2, 2), 8)
            return optimal_threads
        except Exception:
            return 4

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text using simple heuristics."""
        if not text:
            return 0

        # Method 1: Word-based estimate (English: ~1.3 tokens per word)
        word_count = len(text.split())
        word_estimate = int(word_count * 1.5)  # Conservative

        # Method 2: Character-based estimate (English: ~4 chars per token)
        char_count = len(text)
        char_estimate = int(char_count / 3)  # Conservative

        # Method 3: Whitespace and punctuation based
        words = len(re.findall(r"\S+", text))
        punctuation = len(re.findall(r"[.,!?;:]", text))
        special_chars = len(re.findall(r"[^\w\s.,!?;:]", text))
        heuristic_estimate = words + punctuation + special_chars

        # Take the maximum for conservative estimation
        return max(word_estimate, char_estimate, heuristic_estimate)

    def _split_text_if_needed(self, text: str) -> List[str]:
        """Split text into chunks if it exceeds context length."""
        if not text:
            return []

        # Estimate tokens for the full text
        estimated_tokens = self._estimate_tokens(text)

        # If within limits, return as-is
        if estimated_tokens <= self.max_context_tokens:
            return [text]

        # Split the text
        chunks = self.text_splitter.split_text(text)

        # Validate each chunk and further split if necessary
        final_chunks = []
        for chunk in chunks:
            chunk_tokens = self._estimate_tokens(chunk)

            if chunk_tokens <= self.max_context_tokens:
                final_chunks.append(chunk)
            else:
                # If still too long, do aggressive character-based splitting
                # Calculate safe character limit
                safe_char_limit = self.max_context_tokens * 3  # Very conservative

                # Split by characters with word boundaries
                words = chunk.split()
                current_chunk = ""

                for word in words:
                    test_chunk = f"{current_chunk} {word}".strip()

                    if len(test_chunk) <= safe_char_limit:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            final_chunks.append(current_chunk)
                        current_chunk = word

                if current_chunk:
                    final_chunks.append(current_chunk)

        return final_chunks

    def _process_texts_with_splitting(
        self, texts: List[str]
    ) -> Tuple[List[str], List[int]]:
        """Process texts with automatic splitting and return mapping information."""
        processed_chunks = []
        chunk_counts = []

        for text in texts:
            chunks = self._split_text_if_needed(text)
            processed_chunks.extend(chunks)
            chunk_counts.append(len(chunks))

        return processed_chunks, chunk_counts

    def _aggregate_embeddings(
        self,
        embeddings: List[List[float]],
        chunk_counts: List[int],
        aggregation_method: str = "mean",
    ) -> List[List[float]]:
        """Aggregate embeddings from split chunks back to original text count."""
        if not embeddings or not chunk_counts:
            return []

        aggregated = []
        start_idx = 0

        for chunk_count in chunk_counts:
            if chunk_count == 1:
                # Single chunk, use as-is
                aggregated.append(embeddings[start_idx])
            else:
                # Multiple chunks, aggregate
                chunk_embeddings = embeddings[start_idx : start_idx + chunk_count]

                if aggregation_method == "mean":
                    # Average the embeddings
                    aggregated_emb = np.mean(chunk_embeddings, axis=0).tolist()
                elif aggregation_method == "max":
                    # Element-wise maximum
                    aggregated_emb = np.max(chunk_embeddings, axis=0).tolist()
                elif aggregation_method == "first":
                    # Use first chunk
                    aggregated_emb = chunk_embeddings[0]
                elif aggregation_method == "last":
                    # Use last chunk
                    aggregated_emb = chunk_embeddings[-1]
                else:
                    # Default to mean
                    aggregated_emb = np.mean(chunk_embeddings, axis=0).tolist()

                aggregated.append(aggregated_emb)

            start_idx += chunk_count

        return aggregated

    async def _generate_embeddings_with_splitting(
        self, texts: List[str], aggregation_method: str = "mean"
    ) -> List[List[float]]:
        """Generate embeddings with automatic text splitting and aggregation."""
        if not texts:
            return []

        try:
            # Process texts with splitting
            processed_chunks, chunk_counts = self._process_texts_with_splitting(texts)

            if not processed_chunks:
                return [[0.0] * self.embedding_dim for _ in texts]

            # Generate embeddings for all chunks using LangChain
            chunk_embeddings = self.llm.embed_documents(processed_chunks)

            if not chunk_embeddings:
                return [[0.0] * self.embedding_dim for _ in texts]

            # Aggregate chunks back to original texts
            aggregated_embeddings = self._aggregate_embeddings(
                chunk_embeddings, chunk_counts, aggregation_method
            )

            return aggregated_embeddings

        except Exception as e:
            self.logger.error(f"Error generating embeddings with splitting: {e}")
            return [[0.0] * self.embedding_dim for _ in texts]

    def _create_system_prompt(self) -> str:
        """Stub implementation - embedding pipelines don't use system prompts."""
        return ""
