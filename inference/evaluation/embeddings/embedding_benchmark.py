"""
Benchmark class for evaluating embedding models.
"""

import json
import logging
import numpy as np
import os
from typing import Dict, List, Any, Optional

from ..base.benchmark_base import BenchmarkBase
from ..base.result_types import BenchmarkResult

logger = logging.getLogger(__name__)


class EmbeddingBenchmark(BenchmarkBase):
    """
    Benchmark for evaluating embedding models.

    This benchmark evaluates the quality of embeddings by measuring similarity between
    pairs of text and checking if the similarity falls within expected ranges.
    """

    def __init__(self, data_dir: str = "./benchmark_data/embeddings"):
        """Initialize the embedding benchmark.

        Args:
            data_dir: Directory where embedding test data is stored
        """
        super().__init__(
            name="embedding_benchmark",
            description="Evaluates embedding quality through similarity testing",
        )
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Default evaluation sets if none provided
        self.default_eval_set = [
            {
                "query": "What is machine learning?",
                "documents": [
                    "Machine learning is a branch of artificial intelligence.",
                    "Deep learning is a subset of machine learning.",
                    "Quantum computing uses quantum mechanics principles.",
                ],
                "expected_similarities": [
                    {
                        "text": "Machine learning is a branch of artificial intelligence.",
                        "min": 0.7,
                        "max": 1.0,
                    },
                    {
                        "text": "Deep learning is a subset of machine learning.",
                        "min": 0.6,
                        "max": 1.0,
                    },
                    {
                        "text": "Quantum computing uses quantum mechanics principles.",
                        "min": 0.1,
                        "max": 0.5,
                    },
                ],
            },
            {
                "query": "How do neural networks work?",
                "documents": [
                    "Neural networks consist of interconnected nodes.",
                    "Deep learning uses multiple layers of neural networks.",
                    "Reinforcement learning is about training agents through rewards.",
                ],
                "expected_similarities": [
                    {
                        "text": "Neural networks consist of interconnected nodes.",
                        "min": 0.7,
                        "max": 1.0,
                    },
                    {
                        "text": "Deep learning uses multiple layers of neural networks.",
                        "min": 0.6,
                        "max": 1.0,
                    },
                    {
                        "text": "Reinforcement learning is about training agents through rewards.",
                        "min": 0.1,
                        "max": 0.6,
                    },
                ],
            },
        ]

    def get_sample_questions(self) -> List[Dict[str, Any]]:
        """Return sample questions for the embedding benchmark."""
        return self.default_eval_set

    def load_test_data(self, data_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load test data from file or use default if not available.

        Args:
            data_path: Path to test data file

        Returns:
            List of test cases
        """
        if data_path and os.path.exists(data_path):
            try:
                with open(data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.logger.info(
                        f"Loaded {len(data['test_cases'])} test cases from {data_path}"
                    )
                    return data["test_cases"]
            except Exception as e:
                self.logger.error(f"Error loading test data: {str(e)}")

        # Use default eval set if no data file provided or loading failed
        self.logger.info(
            f"Using default evaluation set with {len(self.default_eval_set)} test cases"
        )
        return self.default_eval_set

    def cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            v1: First vector
            v2: Second vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        # Ensure vectors are 1D arrays
        vec1 = np.array(v1).flatten()
        vec2 = np.array(v2).flatten()

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        # Prevent division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def run(
        self,
        model_id: str,
        num_samples: int = 10,
        dataset_path: Optional[str] = None,
        matryoshka_dim: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        Run the embedding benchmark on the specified model.

        Args:
            model_id: Path to the embedding model file
            num_samples: Maximum number of test samples to use
            dataset_path: Optional path to test data
            matryoshka_dim: Optional dimension to truncate embeddings to (e.g., 256)

        Returns:
            BenchmarkResult containing evaluation metrics
        """
        from .extractor import EmbeddingExtractor

        # Set up result structure
        all_results = []
        total_cases = 0
        passed_cases = 0

        # Load test data
        test_data = self.load_test_data(dataset_path)[:num_samples]

        # Initialize extractor
        extractor = EmbeddingExtractor()

        for test_case in test_data:
            # Get query text
            query = test_case["query"]

            # Get query embedding
            try:
                query_embedding = extractor.extract_embedding(
                    text=query,
                    model_path=model_id,
                    is_query=True,
                    matryoshka_dim=matryoshka_dim,
                )

                # Process each document
                document_results = []

                for i, doc in enumerate(test_case["documents"]):
                    # Get document embedding
                    doc_embedding = extractor.extract_embedding(
                        text=doc,
                        model_path=model_id,
                        is_query=False,
                        matryoshka_dim=matryoshka_dim,
                    )

                    # Calculate similarity
                    similarity = self.cosine_similarity(query_embedding, doc_embedding)

                    # Check if similarity is within expected range
                    expected = test_case["expected_similarities"][i]
                    min_sim = expected.get("min", 0.0)
                    max_sim = expected.get("max", 1.0)
                    passed = min_sim <= similarity <= max_sim

                    document_results.append(
                        {
                            "document": doc,
                            "similarity": similarity,
                            "expected_min": min_sim,
                            "expected_max": max_sim,
                            "passed": passed,
                        }
                    )

                    total_cases += 1
                    if passed:
                        passed_cases += 1

                all_results.append({"query": query, "documents": document_results})

            except Exception as e:
                self.logger.error(
                    f"Error running embedding benchmark for {model_id}: {str(e)}"
                )
                all_results.append({"query": query, "error": str(e)})

        # Calculate overall accuracy
        accuracy = passed_cases / total_cases if total_cases > 0 else 0.0

        # Create benchmark result
        return BenchmarkResult(
            score=accuracy * 100,  # Convert to percentage
            total_questions=total_cases,
            correct_answers=passed_cases,
            detailed_results=all_results,
            metadata={
                "model": model_id,
                "matryoshka_dim": matryoshka_dim,
                "total_cases": total_cases,
                "passed_cases": passed_cases,
            },
        )
