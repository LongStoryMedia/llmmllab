"""
Example script to demonstrate how to use the embedding evaluation framework.

This script shows how to use the Nomic Embed Text model to generate embeddings
and evaluate them using the embedding benchmark.
"""

import argparse
import os
import sys

# Add the parent directory to the path to allow importing from sibling packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from evaluation.embeddings.adapter import EmbeddingModelAdapter
from evaluation.embeddings.extractor import EmbeddingExtractor
from evaluation.embeddings.utils import EmbeddingUtils
from evaluation.embeddings.embedding_benchmark import EmbeddingBenchmark
from utils.logging import llmmllogger


def setup_logging():
    """Set up logging configuration."""
    # Logging is now handled by llmmllogger
    pass


def main():
    """Run the embedding evaluation example."""
    parser = argparse.ArgumentParser(description="Run embedding evaluation example")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the Nomic Embed Text GGUF model file",
    )
    parser.add_argument(
        "--matryoshka_dim",
        type=int,
        choices=[256, 512, 768],
        default=None,
        help="Optional Matryoshka dimension to truncate embeddings to (256, 512, or 768)",
    )
    args = parser.parse_args()

    setup_logging()
    logger = llmmllogger.bind(component="embedding_evaluation_example")
    logger.info("Starting embedding evaluation example")

    # Example 1: Generate embeddings using the adapter directly
    logger.info("Example 1: Generate embeddings directly")
    adapter = EmbeddingModelAdapter()

    texts = [
        "What is machine learning?",
        "Machine learning is a branch of artificial intelligence.",
    ]

    embeddings = adapter.generate_embeddings(
        texts=texts, model_path=args.model_path, matryoshka_dim=args.matryoshka_dim
    )

    logger.info(f"Generated {len(embeddings)} embeddings")
    logger.info(f"First embedding dimension: {len(embeddings[0])}")

    # Example 2: Calculate similarity between embeddings
    logger.info("\nExample 2: Calculate similarity between embeddings")
    utils = EmbeddingUtils()
    similarity = utils.cosine_similarity(embeddings[0], embeddings[1])
    logger.info(f"Similarity between query and document: {similarity:.4f}")

    # Example 3: Use the embedding extractor
    logger.info("\nExample 3: Use the embedding extractor")
    extractor = EmbeddingExtractor()
    query_embedding = extractor.extract_embedding(
        text="What is machine learning?",
        model_path=args.model_path,
        is_query=True,
        matryoshka_dim=args.matryoshka_dim,
    )
    logger.info(f"Query embedding dimension: {len(query_embedding)}")

    # Example 4: Run the benchmark
    logger.info("\nExample 4: Run the embedding benchmark")
    benchmark = EmbeddingBenchmark()
    result = benchmark.run(model_id=args.model_path, matryoshka_dim=args.matryoshka_dim)

    logger.info(f"Benchmark score: {result.score:.2f}%")
    logger.info(f"Correct answers: {result.correct_answers}/{result.total_questions}")

    logger.info("Embedding evaluation example completed")


if __name__ == "__main__":
    main()
