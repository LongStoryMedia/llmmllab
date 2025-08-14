#!/usr/bin/env python
"""
Example script showing how to evaluate the Nomic Embed Text v2 model.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path to make imports work
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluation.embeddings.embedding_benchmark import EmbeddingBenchmark
from evaluation.embeddings.extractor import EmbeddingExtractor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to run the embedding evaluation.
    """
    parser = argparse.ArgumentParser(description="Evaluate Nomic Embed Text v2 Model")
    parser.add_argument(
        "--model",
        type=str,
        default="nomic-embed-text-v2",
        help="Model ID for the embedding model",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/app/evaluation/embeddings/data",
        help="Directory containing test data",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="basic_embedding_test.json",
        help="Test file to use for evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="embedding_evaluation_results.json",
        help="Output file to save results",
    )
    parser.add_argument(
        "--matryoshka-dim",
        type=int,
        choices=[256, 512, 768],
        help="Matryoshka dimension for truncated embeddings",
    )
    args = parser.parse_args()

    # Ensure data directory exists
    os.makedirs(args.data_dir, exist_ok=True)

    # Set up full path to test file
    test_file_path = os.path.join(args.data_dir, args.test_file)

    print(f"Using test file: {test_file_path}\ncontent: {open(test_file_path).read()}")

    # Initialize benchmark
    benchmark = EmbeddingBenchmark(data_dir=args.data_dir)

    print(f"Running embedding evaluation for model: {args.model}")
    print(f"Using test data: {test_file_path}")
    if args.matryoshka_dim:
        print(f"Matryoshka dimension: {args.matryoshka_dim}")

    # Run benchmark
    result = benchmark.run(
        model_id=args.model,
        dataset_path=test_file_path,
        matryoshka_dim=args.matryoshka_dim,
    )

    # Display results
    print("\n===== Evaluation Results =====")
    print(f"Model: {args.model}")
    print(f"Score: {result.score:.2f}%")
    print(
        f"Passed Cases: {result.metadata['passed_cases']} / {result.metadata['total_cases']}"
    )

    # Save results
    if args.output:
        # Convert BenchmarkResult to dictionary manually since to_dict() doesn't exist
        result_dict = {
            "score": result.score,
            "total_questions": result.total_questions,
            "correct_answers": result.correct_answers,
            "detailed_results": result.detailed_results,
            "metadata": result.metadata,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
