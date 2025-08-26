"""
Refactored re-ranking pipeline for Qwen3-Reranker-0.6B model.

This pipeline implements the Qwen3-Reranker-0.6B model using the new
BasePipelineDual architecture for ranking text passages by relevance.
"""

import datetime
import logging
import os
from typing import List, Optional, Tuple, AsyncGenerator
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
import torch

from models import (
    Model,
    Message,
    ModelProfile,
)
from ..base_dual_pipeline import TextPipeline
from ..helpers import extract_message_text


logger = logging.getLogger(__name__)


class Qwen3RerankerPipe(TextPipeline):
    """
    Refactored pipeline for Qwen3-Reranker-0.6B model using BasePipelineDual architecture.

    Features:
    - Re-ranking passages by relevance to queries
    - 0.6B parameters optimized for ranking tasks
    - Context length up to 8192 tokens
    - Batch processing support
    - Designed for RAG and search applications
    """

    llm: LlamaCpp

    def __init__(self, model: Model, profile: ModelProfile):
        """Initialize the Qwen3 Reranker pipeline."""
        super().__init__(model, profile)

        self._logger = logging.getLogger(__name__)
        self._logger.info("Initializing Qwen3RerankerPipe")

        # Validate model definition
        if not (model.details and model.model):
            raise ValueError(
                "Model definition for Qwen3RerankerPipe must include model path details."
            )

        # Get and validate GGUF file
        gguf_path = self._get_gguf_path()
        self._validate_gguf_file(gguf_path)

        # Initialize model with optimizations
        self._initialize_model(gguf_path)

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

        self._logger.info(
            f"Using GGUF file: {gguf_path} (size: {file_size/1_000_000:.2f} MB)"
        )

    def _initialize_model(self, gguf_path: str) -> None:
        """Initialize the ChatLlamaCpp model for reranking with optimizations."""
        try:
            # Get context size from profile or use default
            context_size = min(self.profile.parameters.num_ctx or 8192, 8192)

            self.llm = LlamaCpp(
                model_path=gguf_path,
                n_ctx=context_size,
                n_gpu_layers=-1,  # Offload all layers to GPU
                n_threads=self._get_optimal_threads(),
                f16_kv=True,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                verbose=os.getenv("LOG_LEVEL", "WARNING").lower() == "debug",
                n_batch=512,  # Optimized batch size
                # Reranking-specific parameters
                streaming=False,  # Reranking doesn't need streaming
                # Stop tokens optimized for relevance scoring
                stop=[
                    "\n",
                    ".",
                    "Yes",
                    "No",
                    "True",
                    "False",
                    "Relevant",
                    "Irrelevant",
                ],
                n_parts=-1,
                seed=self.profile.parameters.seed or -1,
                logits_all=False,
                vocab_only=False,
                use_mlock=False,  # Better memory management
                suffix="",
                logprobs=0,
                # Optimized generation parameters
                temperature=self.profile.parameters.temperature or 0.0,
                max_tokens=self.profile.parameters.num_predict or 20,
                top_p=self.profile.parameters.top_p or 0.8,
                top_k=self.profile.parameters.top_k or 20,
                repeat_penalty=self.profile.parameters.repeat_penalty or 1.05,
            )

            self._logger.info(
                f"Qwen3 Reranker model '{self.model.name}' loaded successfully with context size {context_size}"
            )
        except Exception as e:
            self._logger.error(
                f"Error initializing {self.__class__.__name__}: {str(e)}"
            )
            raise

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

    def _create_ranking_prompt(self, query: str, passage: str) -> str:
        """Create a standardized prompt for relevance scoring."""
        return f"""Query: {query}
Passage: {passage}
Is this passage relevant to the query? Answer with only "Yes" or "No":"""

    async def run(self, messages: List[Message]) -> AsyncGenerator[str, None]:
        """
        Process messages to generate ranking scores.

        Expected input format:
        - First message contains the query
        - Subsequent messages contain passages to rank

        Args:
            messages: List of messages (query + passages to rank)

        Yields:
            str: Ranking results as formatted text
        """
        start_time = datetime.datetime.now(datetime.timezone.utc)

        try:
            # Extract query and passages from messages
            query, passages = self._extract_query_and_passages(messages)

            if not query:
                yield "Error: No query provided"
                return

            if not passages:
                yield "Error: No passages provided for ranking"
                return

            self._logger.info(f"Ranking {len(passages)} passages for query")

            # Generate ranking scores
            scores = await self._rank_passages_async(query, passages)

            # Format results
            ranking_text = await self._format_ranking_results(query, passages, scores)

            # Log performance
            duration = (
                datetime.datetime.now(datetime.timezone.utc) - start_time
            ).total_seconds()
            self._logger.debug(f"Ranked {len(passages)} passages in {duration:.2f}s")

            yield ranking_text

        except Exception as e:
            self._logger.error(f"Error in reranking: {e}")
            yield f"Error in reranking: {str(e)}"

    def _extract_query_and_passages(
        self, messages: List[Message]
    ) -> Tuple[str, List[str]]:
        """Extract query and passages from input messages."""
        if len(messages) < 2:
            return "", []

        # First message is the query
        query = extract_message_text(messages[0])

        # Remaining messages are passages
        passages = []
        for message in messages[1:]:
            passage = extract_message_text(message)
            if passage:
                passages.append(passage)

        return query, passages

    async def _rank_passages_async(
        self, query: str, passages: List[str]
    ) -> List[Tuple[int, float]]:
        """
        Rank passages asynchronously for better performance.

        Args:
            query: Query text
            passages: List of passage texts

        Returns:
            List of (original_index, relevance_score) tuples
        """
        scores = []

        for i, passage in enumerate(passages):
            try:
                # Create ranking prompt
                prompt = self._create_ranking_prompt(query, passage)

                # Get relevance score from model
                response = await self.llm.ainvoke(prompt)
                score = self._extract_relevance_score(response)

                scores.append((i, score))
                self._logger.debug(f"Passage {i}: score={score:.4f}")

            except Exception as e:
                self._logger.error(f"Error processing passage {i}: {e}")
                scores.append((i, 0.0))  # Default low score for errors

        return scores

    def _extract_relevance_score(self, response_text: str) -> float:
        """
        Extract relevance score from model response.

        Args:
            response_text: Raw response from the model

        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not response_text:
            return 0.5

        text = response_text.strip().lower()

        # Direct matches
        if text.startswith("yes") or "yes" in text:
            return 1.0
        if text.startswith("no") or "no" in text:
            return 0.0
        if "relevant" in text:
            return 0.9
        if "irrelevant" in text or "not relevant" in text:
            return 0.1

        # Fallback to neutral score
        return 0.5

    async def _format_ranking_results(
        self, query: str, passages: List[str], scores: List[Tuple[int, float]]
    ) -> str:
        """Format ranking results into readable text."""
        # Sort by score (descending)
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

        result_lines = [
            f"Re-ranked {len(passages)} passages for query: '{query[:100]}{'...' if len(query) > 100 else ''}'",
            "",
            "Rankings (rank: original_index - score - preview):",
        ]

        for rank, (orig_idx, score) in enumerate(sorted_scores, 1):
            passage_preview = (
                passages[orig_idx][:150] + "..."
                if len(passages[orig_idx]) > 150
                else passages[orig_idx]
            )
            result_lines.append(f"{rank}: {orig_idx} - {score:.4f} - {passage_preview}")

        return "\n".join(result_lines)

    async def rank_passages(
        self, query: str, passages: List[str], top_k: Optional[int] = None
    ) -> List[Tuple[int, float, str]]:
        """
        Convenience method for direct passage ranking.

        Args:
            query: Query text
            passages: List of passage texts to rank
            top_k: Optional limit on results

        Returns:
            List of (original_index, relevance_score, passage_text) tuples
        """
        # Convert to messages
        query_message = self.create_user_message(query)
        passage_messages = [self.create_user_message(passage) for passage in passages]
        messages = [query_message] + passage_messages

        # Get rankings
        async for result_text in self.run(messages):
            # Parse the ranking results (this is a simplified approach)
            # In a real implementation, you might want to return structured data
            scores = await self._rank_passages_async(query, passages)

            # Convert to the expected format
            results = []
            for orig_idx, score in scores:
                results.append((orig_idx, score, passages[orig_idx]))

            # Sort by score (descending)
            results.sort(key=lambda x: x[1], reverse=True)

            return results[:top_k] if top_k else results

        # Fallback
        return [
            (i, 0.0, passage)
            for i, passage in enumerate(passages[:top_k] if top_k else passages)
        ]

    async def compute_pairwise_scores(
        self, query: str, passages: List[str]
    ) -> List[List[float]]:
        """
        Compute pairwise relevance scores for advanced ranking.

        Args:
            query: Query text
            passages: List of passage texts

        Returns:
            Matrix of pairwise relevance scores
        """
        n = len(passages)
        score_matrix = [[0.0 for _ in range(n)] for _ in range(n)]

        # Get individual relevance scores first
        individual_scores = await self._rank_passages_async(query, passages)
        score_dict = {idx: score for idx, score in individual_scores}

        # Fill diagonal with individual scores
        for i in range(n):
            score_matrix[i][i] = score_dict.get(i, 0.0)

        # For pairwise comparison, we can use the individual scores as an approximation
        # In a more sophisticated implementation, you might compare passages directly
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Simple approximation: average of individual scores
                    score_i = score_dict.get(i, 0.0)
                    score_j = score_dict.get(j, 0.0)
                    score_matrix[i][j] = (score_i + score_j) / 2.0

        return score_matrix

    async def health_check(self) -> bool:
        """
        Perform a health check on the reranker pipeline.

        Returns:
            bool: True if the pipeline is healthy, False otherwise
        """
        try:
            # Test with a simple query and passage
            test_query = "What is machine learning?"
            test_passage = "Machine learning is a subset of artificial intelligence."

            # Try to rank the passage
            results = await self.rank_passages(test_query, [test_passage])

            return len(results) > 0 and isinstance(results[0][1], (int, float))

        except Exception as e:
            self._logger.error(f"Health check failed: {e}")
            return False

    def get_model_info(self) -> dict:
        """Get information about the loaded reranker model."""
        return {
            "model_name": self.model.name,
            "model_id": self.model.id,
            "model_path": self.model.details.gguf_file if self.model.details else None,
            "context_size": getattr(self.llm, "n_ctx", None),
            "task": "text_reranking",
            "max_tokens": getattr(self.llm, "max_tokens", None),
        }

    def __del__(self) -> None:
        """Clean up resources with enhanced error handling."""
        try:
            model_name = (
                getattr(self.model, "name", "unknown")
                if hasattr(self, "model")
                else "unknown"
            )
            self._logger.info(f"Qwen3RerankerPipe for {model_name}: Cleanup initiated")

            if hasattr(self, "llm") and self.llm is not None:
                del self.llm

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        except Exception as e:
            logger.error(f"Error cleaning up Qwen3RerankerPipe: {e}")
