"""
Direct Qwen3-Embedding-0.6B implementation using llama_cpp.Llama.
"""

import logging
import multiprocessing
import os
import re
from typing import List, Optional

import llama_cpp
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import Model, ModelProfile


class Qwen3Embeddings(Embeddings):
    """
    Direct Qwen3-Embedding-0.6B implementation using llama_cpp.Llama.

    Features:
    - Direct llama_cpp.Llama usage with embed() method
    - 1024-dimensional embeddings
    - 8192 token context window with automatic text splitting
    - Optimized for multilingual text understanding
    """

    def __init__(self, model: Model, profile: ModelProfile):
        """Initialize the Qwen3 embeddings with direct Llama instance."""
        self.model = model
        self.profile = profile
        self._logger = logging.getLogger(self.__class__.__name__)
        self.llama_instance: Optional[llama_cpp.Llama] = None
        self.embedding_dim = 1024  # Qwen3 embedding dimension
        self.max_context_tokens = 8192
        self.text_splitter: Optional[RecursiveCharacterTextSplitter] = None

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

        # Validate file size
        file_size = os.path.getsize(gguf_path)
        if file_size < 1_000_000:  # Less than 1MB is suspicious
            raise ValueError(f"GGUF file is too small ({file_size} bytes): {gguf_path}")

        try:
            context_size = min(self.profile.parameters.num_ctx or 8192, 8192)

            self.llama_instance = llama_cpp.Llama(
                model_path=gguf_path,
                n_ctx=context_size,
                n_gpu_layers=-1,  # Offload all layers to GPU
                n_threads=self._get_optimal_threads(),
                embedding=True,  # Enable embedding mode
                verbose=os.getenv("LOG_LEVEL", "WARNING").lower() == "debug",
                n_batch=512,
                use_mlock=False,
                f16_kv=True,
                split_mode=llama_cpp.LLAMA_SPLIT_MODE_LAYER,
                # main_gpu=0,
                tensor_split=None,
                vocab_only=False,
                use_mmap=True,
                kv_overrides=None,
                # Context Params
                seed=self.profile.parameters.seed or llama_cpp.LLAMA_DEFAULT_SEED,
                n_ubatch=512,
                temperature=self.profile.parameters.temperature or 0.7,
                top_p=self.profile.parameters.top_p or 0.8,
                top_k=self.profile.parameters.top_k or 20,
                repeat_penalty=self.profile.parameters.repeat_penalty or 1.05,
                flash_attn=getattr(self.profile.parameters, "flash_attention", True),
                chat_format=None,  # Default chat format, can be overridden
                n_threads_batch=None,
                rope_scaling_type=None,
                pooling_type=llama_cpp.LLAMA_POOLING_TYPE_UNSPECIFIED,
                rope_freq_base=0.0,
                rope_freq_scale=0.0,
                yarn_ext_factor=-1.0,
                yarn_attn_factor=1.0,
                yarn_beta_fast=32.0,
                yarn_beta_slow=1.0,
                yarn_orig_ctx=0,
                offload_kqv=False,  # Disable offload for better performance unless needed
                op_offload=None,
                swa_full=None,
                # Sampling Params
                no_perf=False,
                last_n_tokens_size=64,
                # LoRA Params
                lora_base=None,
                lora_scale=1.0,
                lora_path=None,
                # Backend Params
                numa=False,
                chat_handler=None,
                # Speculative Decoding
                draft_model=None,
                # Tokenizer Override
                tokenizer=None,
                # KV cache quantization
                type_k=None,
                type_v=None,
                # Misc
                spm_infill=False,
            )

            self._logger.info(f"Qwen3 embedding model initialized from: {gguf_path}")
            return self.llama_instance

        except Exception as e:
            self._logger.error(f"Failed to initialize Qwen3 embeddings: {e}")
            raise

    def _init_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Initialize text splitter for handling long texts."""
        if self.text_splitter is None:
            # Use conservative character-to-token ratio (3:1) to avoid exceeding limits
            max_chunk_chars = self.max_context_tokens * 3

            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_chunk_chars,
                chunk_overlap=max_chunk_chars // 10,  # 10% overlap
                separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
                keep_separator=True,
            )
        return self.text_splitter

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text using simple heuristics."""
        if not text:
            return 0

        # Conservative estimates
        word_count = len(text.split())
        word_estimate = int(word_count * 1.5)  # Conservative

        char_count = len(text)
        char_estimate = int(char_count / 3)  # Conservative

        words = len(re.findall(r"\S+", text))
        punctuation = len(re.findall(r"[.,!?;:]", text))
        special_chars = len(re.findall(r"[^\w\s.,!?;:]", text))
        heuristic_estimate = words + punctuation + special_chars

        return max(word_estimate, char_estimate, heuristic_estimate)

    def _split_text_if_needed(self, text: str) -> List[str]:
        """Split text into chunks if it exceeds context length."""
        if not text:
            return []

        estimated_tokens = self._estimate_tokens(text)
        if estimated_tokens <= self.max_context_tokens:
            return [text]

        # Initialize text splitter if needed
        splitter = self._init_text_splitter()
        chunks = splitter.split_text(text)

        # Validate each chunk
        final_chunks = []
        for chunk in chunks:
            chunk_tokens = self._estimate_tokens(chunk)
            if chunk_tokens <= self.max_context_tokens:
                final_chunks.append(chunk)
            else:
                # Further split if still too large
                words = chunk.split()
                chunk_size = len(words) // 2
                while chunk_size > 0:
                    sub_chunk = " ".join(words[:chunk_size])
                    if self._estimate_tokens(sub_chunk) <= self.max_context_tokens:
                        final_chunks.append(sub_chunk)
                        words = words[chunk_size:]
                        chunk_size = len(words) // 2
                    else:
                        chunk_size //= 2

        return final_chunks

    def _aggregate_embeddings(self, embeddings: List[List[float]]) -> List[float]:
        """Aggregate multiple embeddings using mean pooling."""
        if not embeddings:
            return [0.0] * self.embedding_dim

        if len(embeddings) == 1:
            return embeddings[0]

        # Mean pooling
        aggregated = [0.0] * len(embeddings[0])
        for embedding in embeddings:
            for i, val in enumerate(embedding):
                aggregated[i] += val

        return [val / len(embeddings) for val in aggregated]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        if not texts:
            return []

        # Initialize model if needed
        if self.llama_instance is None:
            self.llama_instance = self._initialize_llama()

        embeddings = []
        for text in texts:
            # Split text if needed
            chunks = self._split_text_if_needed(text)

            chunk_embeddings = []
            for chunk in chunks:
                embedding_result = self.llama_instance.embed(chunk)

                # Extract embedding vector from response
                if isinstance(embedding_result, list) and len(embedding_result) > 0:
                    if isinstance(embedding_result[0], list):
                        chunk_embeddings.append(embedding_result[0])
                    elif isinstance(embedding_result[0], (int, float)):
                        chunk_embeddings.append(
                            [
                                float(x) if isinstance(x, (int, float)) else 0.0
                                for x in embedding_result
                            ]
                        )
                    else:
                        chunk_embeddings.append([0.0] * self.embedding_dim)
                else:
                    chunk_embeddings.append([0.0] * self.embedding_dim)

            # Aggregate embeddings if multiple chunks
            final_embedding = self._aggregate_embeddings(chunk_embeddings)
            embeddings.append(final_embedding)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        # Initialize model if needed
        if self.llama_instance is None:
            self.llama_instance = self._initialize_llama()

        # Split text if needed
        chunks = self._split_text_if_needed(text)

        chunk_embeddings = []
        for chunk in chunks:
            embedding_result = self.llama_instance.embed(chunk)

            # Extract embedding vector from response
            if isinstance(embedding_result, list) and len(embedding_result) > 0:
                if isinstance(embedding_result[0], list):
                    chunk_embeddings.append(embedding_result[0])
                elif isinstance(embedding_result[0], (int, float)):
                    chunk_embeddings.append(embedding_result)
                else:
                    chunk_embeddings.append([0.0] * self.embedding_dim)
            else:
                chunk_embeddings.append([0.0] * self.embedding_dim)

        # Aggregate embeddings if multiple chunks
        return self._aggregate_embeddings(chunk_embeddings)
