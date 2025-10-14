"""
Similarity Ranker Node for LangGraph workflows.
Ranks embeddings by similarity to a query embedding.
"""

from typing import List, Optional

import numpy as np

from composer.graph.state import WorkflowState
from utils.logging import llmmllogger


class SimilarityRankerNode:
    """
    Node for ranking embeddings by cosine similarity to a query embedding.

    Takes query embedding and candidate embeddings from workflow state,
    computes similarity scores, and returns ranked results.
    """

    def __init__(self):
        """Initialize similarity ranker node."""
        self.logger = llmmllogger.logger.bind(component="SimilarityRankerNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Rank candidate embeddings by similarity to query embedding.

        Args:
            state: Current workflow state with query_embedding and candidate_embeddings

        Returns:
            Updated workflow state with ranked_similarities in execution_metadata
        """
        try:
            # Get query embedding and candidates from state
            if not state.embedding:
                self.logger.warning("No query embedding found in state")
                return state

            query_embedding = state.embedding
            candidate_embeddings = state.unranked_retrievals

            if not query_embedding:
                self.logger.warning("No query embedding found in state")
                return state

            if not candidate_embeddings:
                self.logger.warning("No candidate embeddings found in state")
                return state

            self.logger.info(
                "Ranking embeddings by similarity",
                query_dimensions=len(query_embedding),
                candidate_count=len(candidate_embeddings),
            )

            # Calculate similarities
            similarities = []
            for i, candidate in enumerate(candidate_embeddings):
                similarity = self._compute_cosine_similarity(query_embedding, candidate)
                similarities.append((i, similarity))

            # Sort by similarity score (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Store ranked results in state
            state.ranked_retrievals = similarities

            self.logger.info(
                "Successfully ranked embeddings",
                total_candidates=len(similarities),
                top_similarity=similarities[0][1] if similarities else 0.0,
            )

            return state

        except Exception as e:
            self.logger.error("Similarity ranking failed", error=str(e))
            # Don't raise - add error to state and continue workflow
            state.error_details.append(f"Similarity ranking failed: {str(e)}")
            return state

    def _compute_cosine_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embedding vectors.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            self.logger.error(f"Cosine similarity computation failed: {e}")
            return 0.0

    async def rank_memories_by_similarity(self, state: WorkflowState) -> WorkflowState:
        """
        Rank memories by similarity to query, maintaining memory metadata.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with ranked_memories
        """
        try:
            # Get query embedding and retrieved memories from state
            query_embedding = state.embedding
            retrieved_memories = state.retrieved_memories

            if not query_embedding or not retrieved_memories:
                self.logger.info("No query embedding or memories to rank")
                return state

            self.logger.info(
                "Ranking memories by similarity",
                memory_count=len(retrieved_memories),
                query_dimensions=len(query_embedding),
            )

            # Calculate similarity for each memory and re-rank
            memory_similarities = []
            for memory in retrieved_memories:
                # Assume memory has an embedding attribute or compute from fragments
                memory_embedding = self._extract_memory_embedding(memory)
                if memory_embedding:
                    similarity = self._compute_cosine_similarity(
                        query_embedding, memory_embedding
                    )
                    memory_similarities.append((memory, similarity))

            # Sort by similarity (highest first)
            memory_similarities.sort(key=lambda x: x[1], reverse=True)

            # Update memories with new similarity scores and store back
            ranked_memories = []
            for memory, similarity in memory_similarities:
                memory.similarity = similarity  # Update similarity score
                ranked_memories.append(memory)

            state.retrieved_memories = ranked_memories

            self.logger.info(
                "Successfully ranked memories",
                total_memories=len(ranked_memories),
                top_similarity=(
                    memory_similarities[0][1] if memory_similarities else 0.0
                ),
            )

            return state

        except Exception as e:
            self.logger.error("Memory ranking failed", error=str(e))
            # Don't raise - add error to state and continue workflow
            state.error_details.append(f"Memory ranking failed: {str(e)}")
            return state

    def _extract_memory_embedding(self, memory) -> Optional[List[float]]:
        """
        Extract embedding from memory object.

        Args:
            memory: Memory object

        Returns:
            Embedding vector or None if not available
        """
        # Try different ways to get embedding from memory
        if hasattr(memory, "embedding") and memory.embedding:
            return memory.embedding
        elif hasattr(memory, "fragments") and memory.fragments:
            # Use first fragment's embedding if available
            first_fragment = memory.fragments[0]
            if hasattr(first_fragment, "embedding") and first_fragment.embedding:
                return first_fragment.embedding

        return None
