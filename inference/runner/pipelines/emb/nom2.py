"""
Enhanced Nomic Embed Text v2 pipeline with token counting and text splitting.
"""

import datetime
import logging
import os
import re
from typing import List, Optional, Union, AsyncGenerator, cast, Tuple, Any
import torch
import numpy as np
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from models import (
    Model,
    Message,
    ModelProfile,
)
from ..base import EmbeddingPipeline
from utils.message import extract_message_text


logger = logging.getLogger(__name__)


class NomicEmbedTextPipe(EmbeddingPipeline):
    """
    Enhanced pipeline for Nomic Embed Text v2 MoE model with token splitting.

    Features:
    - Multilingual MoE text embedding (305M active, 475M total parameters)
    - Supports ~100 languages with 768-dimensional embeddings
    - Task-specific prefixes: "search_query:" for queries, "search_document:" for documents
    - Maximum sequence length: 512 tokens with automatic text splitting
    - Matryoshka embedding support (256, 512, 768 dimensions)
    - Automatic token counting and text chunking
    """

    llm: LlamaCppEmbeddings

    def __init__(self, model: Model, profile: ModelProfile):
        """Initialize the Nomic Embed Text pipeline."""
        super().__init__(model, profile)

        self._logger = logging.getLogger(__name__)
        self._logger.info("Initializing NomicEmbedTextPipe")

        # Nomic-specific parameters
        self.max_context_tokens = 512
        self.embedding_dim = 768

        # Validate model definition
        if not (model.details and model.model):
            raise ValueError(
                "Model definition for NomicEmbedTextPipe must include model path details."
            )

        # Get and validate GGUF file
        gguf_path = self._get_gguf_path()
        self._validate_gguf_file(gguf_path)

        # Initialize model with optimizations
        self._initialize_model(gguf_path)

        # Initialize text splitter for handling long texts
        self._init_text_splitter()

    # --- BasePipelineCore required methods ---
    async def process_messages(
        self, messages: List[Message], tools: Optional[List[Any]] = None
    ) -> List[List[float]]:
        """Process messages and return embeddings (no tools/graph needed)."""
        # Extract and process texts with prefixes
        texts: List[str] = []
        for message in messages:
            text = extract_message_text(message)
            if text:
                texts.append(self._add_task_prefix(text))

        if not texts:
            return [[0.0] * self.embedding_dim]

        return await self._generate_embeddings_with_splitting(
            texts, aggregation_method="mean"
        )

    def create_graph(self, tools: Optional[List[Any]] = None):  # type: ignore[override]
        """Embeddings pipeline doesn't use LangGraph workflows."""
        raise NotImplementedError("Embedding pipelines do not use LangGraph graphs")

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

        self._logger.debug(
            f"Initialized text splitter with max_chunk_chars={max_chunk_chars}"
        )

    def _get_gguf_path(self) -> str:
        """Get the GGUF file path from model definition."""
        assert self.model
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

        self._logger.info(
            f"Using GGUF file: {gguf_path} (size: {file_size/1_000_000:.2f} MB)"
        )

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

    def _initialize_model(self, gguf_path: str) -> None:
        """Initialize the LangChain LlamaCppEmbeddings model with optimizations."""
        try:
            # Use the same context size as the model's maximum (512 for Nomic)
            context_size = min(self.profile.parameters.num_ctx or 512, 512)

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

            self._logger.info(
                f"Nomic Embed Text model '{self.model.name}' loaded successfully "
                f"(max_tokens: {self.max_context_tokens}, dims: {self.embedding_dim})"
            )
        except Exception as e:
            self._logger.error(
                f"Error initializing {self.__class__.__name__}: {str(e)}"
            )
            raise

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text using simple heuristics.
        """
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
        estimate = max(word_estimate, char_estimate, heuristic_estimate)

        self._logger.debug(
            f"Token estimates - Word: {word_estimate}, Char: {char_estimate}, "
            f"Heuristic: {heuristic_estimate}, Using: {estimate}"
        )

        return estimate

    def _split_text_if_needed(self, text: str) -> List[str]:
        """Split text into chunks if it exceeds context length."""
        if not text:
            return []

        # Estimate tokens for the full text
        estimated_tokens = self._estimate_tokens(text)

        # If within limits, return as-is
        if estimated_tokens <= self.max_context_tokens:
            return [text]

        self._logger.info(
            f"Text exceeds token limit ({estimated_tokens} > {self.max_context_tokens}), splitting..."
        )

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
                self._logger.warning(
                    f"Chunk still too long ({chunk_tokens} tokens), doing aggressive split"
                )

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

        self._logger.info(f"Split text into {len(final_chunks)} chunks")
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

            if len(chunks) > 1:
                self._logger.debug(f"Split text into {len(chunks)} chunks")

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

                self._logger.debug(
                    f"Aggregated {chunk_count} chunks using {aggregation_method} method"
                )

            start_idx += chunk_count

        return aggregated

    def _add_task_prefix(self, text: str, is_query: Optional[bool] = None) -> str:
        """Add appropriate task prefix to text based on content or explicit flag."""
        # Check if text already has a prefix
        if text.startswith(("search_document: ", "search_query: ")):
            return text

        # Determine prefix based on is_query flag or text length heuristic
        if is_query is True:
            prefix = "search_query: "
        elif is_query is False:
            prefix = "search_document: "
        else:
            # Auto-detect: shorter texts are likely queries, longer ones are documents
            prefix = "search_query: " if len(text) < 100 else "search_document: "

        return f"{prefix}{text}"

    async def run(
        self, messages: List[Message]
    ) -> AsyncGenerator[List[List[float]], None]:
        """
        Process messages to generate embeddings with task prefixes and automatic splitting.
        """
        start_time = datetime.datetime.now(datetime.timezone.utc)

        try:
            # Extract and process texts with prefixes
            texts = []
            for message in messages:
                text = extract_message_text(message)
                if text:
                    # Add task prefix (auto-detect query vs document)
                    prefixed_text = self._add_task_prefix(text)
                    texts.append(prefixed_text)

            if not texts:
                self._logger.warning("No text inputs found in messages")
                yield [[0.0] * self.embedding_dim]
                return

            self._logger.info(
                f"Processing {len(texts)} text inputs with task prefixes and splitting"
            )

            # Generate embeddings with splitting
            embeddings = await self._generate_embeddings_with_splitting(
                texts, aggregation_method="mean"
            )

            # Log performance
            duration = (
                datetime.datetime.now(datetime.timezone.utc) - start_time
            ).total_seconds()
            self._logger.debug(
                f"Generated {len(embeddings)} embeddings in {duration:.2f}s "
                f"({len(embeddings[0]) if embeddings else 0} dimensions each)"
            )

            yield embeddings

        except Exception as e:
            self._logger.error(f"Error in embedding generation: {e}")
            # Return zero embeddings as fallback
            yield [
                [0.0] * self.embedding_dim
                for _ in range(len(texts) if "texts" in locals() else 1)
            ]

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

            self._logger.info(
                f"Processing {len(processed_chunks)} chunks from {len(texts)} original texts"
            )

            # Generate embeddings for all chunks using LangChain
            chunk_embeddings = await self.llm.aembed_documents(processed_chunks)

            if not chunk_embeddings:
                self._logger.warning("No embeddings returned from model")
                return [[0.0] * self.embedding_dim for _ in texts]

            # Validate embedding dimensions
            if chunk_embeddings and len(chunk_embeddings[0]) != self.embedding_dim:
                self._logger.warning(
                    f"Unexpected embedding dimension: {len(chunk_embeddings[0])}, expected {self.embedding_dim}"
                )

            # Aggregate chunks back to original texts
            aggregated_embeddings = self._aggregate_embeddings(
                chunk_embeddings, chunk_counts, aggregation_method
            )

            return cast(List[List[float]], aggregated_embeddings)

        except Exception as e:
            self._logger.error(f"Error generating embeddings with splitting: {e}")
            return [[0.0] * self.embedding_dim for _ in texts]

    async def embed_texts(
        self,
        texts: Union[str, List[str]],
        is_query: Optional[bool] = None,
        matryoshka_dim: Optional[int] = None,
        aggregation_method: str = "mean",
    ) -> List[List[float]]:
        """
        Convenience method for direct text embedding with task prefixes and splitting.
        """
        if isinstance(texts, str):
            texts = [texts]

        # Add task prefixes
        processed_texts = [self._add_task_prefix(text, is_query) for text in texts]

        # Generate embeddings with splitting
        embeddings = await self._generate_embeddings_with_splitting(
            processed_texts, aggregation_method
        )

        # Apply Matryoshka truncation if requested
        if matryoshka_dim and matryoshka_dim in [256, 512, 768]:
            self._logger.debug(f"Truncating embeddings to {matryoshka_dim} dimensions")
            embeddings = [emb[:matryoshka_dim] for emb in embeddings]
        elif matryoshka_dim and matryoshka_dim not in [256, 512, 768]:
            self._logger.warning(
                f"Invalid matryoshka_dim: {matryoshka_dim}. Using full {self.embedding_dim} dimensions."
            )

        return embeddings

    async def embed_query(
        self,
        query: str,
        matryoshka_dim: Optional[int] = None,
        aggregation_method: str = "mean",
    ) -> List[float]:
        """Embed a single query with proper task prefix and splitting."""
        embeddings = await self.embed_texts(
            [query],
            is_query=True,
            matryoshka_dim=matryoshka_dim,
            aggregation_method=aggregation_method,
        )
        return (
            embeddings[0]
            if embeddings
            else [0.0] * (matryoshka_dim or self.embedding_dim)
        )

    async def embed_documents(
        self,
        documents: List[str],
        matryoshka_dim: Optional[int] = None,
        aggregation_method: str = "mean",
    ) -> List[List[float]]:
        """Embed multiple documents with proper task prefix and splitting."""
        return await self.embed_texts(
            documents,
            is_query=False,
            matryoshka_dim=matryoshka_dim,
            aggregation_method=aggregation_method,
        )

    async def compute_similarity(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        aggregation_method: str = "mean",
    ) -> List[tuple[int, float, str]]:
        """Compute similarity between query and documents with automatic splitting."""
        try:
            # Generate embeddings with proper task prefixes and splitting
            query_prefixed = self._add_task_prefix(query, is_query=True)
            docs_prefixed = [
                self._add_task_prefix(doc, is_query=False) for doc in documents
            ]

            query_embeddings = await self._generate_embeddings_with_splitting(
                [query_prefixed], aggregation_method
            )
            doc_embeddings = await self._generate_embeddings_with_splitting(
                docs_prefixed, aggregation_method
            )

            if not query_embeddings or not doc_embeddings:
                self._logger.error("Failed to generate embeddings")
                return []

            # Compute cosine similarities
            query_emb = np.array(query_embeddings[0])
            similarities = []

            for i, doc_emb in enumerate(doc_embeddings):
                doc_emb_array = np.array(doc_emb)
                similarity = float(
                    np.dot(query_emb, doc_emb_array)
                )  # Already normalized
                similarities.append((i, similarity, documents[i]))

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)

            return similarities[:top_k] if top_k else similarities

        except Exception as e:
            self._logger.error(f"Error computing similarities: {e}")
            return [
                (i, 0.0, doc)
                for i, doc in enumerate(documents[:top_k] if top_k else documents)
            ]

    def get_token_count_estimate(self, text: str) -> int:
        """Get token count estimate for a text (including any prefixes)."""
        prefixed_text = self._add_task_prefix(text)
        return self._estimate_tokens(prefixed_text)

    def will_text_be_split(self, text: str, is_query: Optional[bool] = None) -> bool:
        """Check if a text will be split due to token limits."""
        prefixed_text = self._add_task_prefix(text, is_query)
        estimated_tokens = self._estimate_tokens(prefixed_text)
        return estimated_tokens > self.max_context_tokens

    def __del__(self) -> None:
        """Clean up resources with enhanced error handling."""
        try:
            model_name = (
                getattr(self.model, "name", "unknown")
                if hasattr(self, "model")
                else "unknown"
            )
            self._logger.info(f"NomicEmbedTextPipe for {model_name}: Cleanup initiated")

            if hasattr(self, "llm") and self.llm is not None:
                del self.llm

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        except Exception as e:
            logger.error(f"Error cleaning up NomicEmbedTextPipe: {e}")
