"""
CLI tool to run embedding model evaluations.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from .embedding_benchmark import EmbeddingBenchmark
from .utils import EmbeddingUtils

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_test_data(file_path: str) -> Dict[str, Any]:
    """Load test data from JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading test data from {file_path}: {str(e)}")
        sys.exit(1)


def save_results(results: Dict[str, Any], output_file: str):
    """Save benchmark results to a file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_file}")


def run_embedding_evaluation(args):
    """Run embedding model evaluation."""
    logger.info(f"Running embedding evaluation for model: {args.model}")

    # Set up benchmark
    benchmark = EmbeddingBenchmark(data_dir=args.data_dir)

    # Load test data if specified
    test_data_path = None
    if args.test_data:
        test_data_path = os.path.join(args.data_dir, args.test_data)
        if not os.path.exists(test_data_path):
            logger.warning(f"Test data file not found: {test_data_path}")
            test_data_path = None

    # Run benchmark
    result = benchmark.run(
        model_id=args.model,
        num_samples=args.samples,
        dataset_path=test_data_path,
        matryoshka_dim=args.matryoshka_dim,
    )

    # Display summary
    print("\n===== Embedding Evaluation Results =====")
    print(f"Model: {args.model}")
    print(f"Score: {result.score:.2f}%")
    print(
        f"Passed Cases: {result.metadata['passed_cases']} / {result.metadata['total_cases']}"
    )
    if args.matryoshka_dim:
        print(f"Matryoshka Dimension: {args.matryoshka_dim}")

    # Save results if output file specified
    if args.output:
        save_results(result.to_dict(), args.output)

    return result


def interactive_mode(args):
    """Run interactive embedding testing mode."""
    logger.info(f"Starting interactive mode with model: {args.model}")

    utils = EmbeddingUtils()

    print("\n===== Interactive Embedding Testing =====")
    print(f"Model: {args.model}")
    if args.matryoshka_dim:
        print(f"Matryoshka Dimension: {args.matryoshka_dim}")

    while True:
        print("\nEnter query text (or 'quit' to exit):")
        query = input("> ")
        if query.lower() in ("quit", "exit", "q"):
            break

        print("Enter documents to compare (one per line, empty line to finish):")
        documents = []
        while True:
            doc = input("> ")
            if not doc:
                break
            documents.append(doc)

        if not documents:
            print("No documents entered. Please try again.")
            continue

        print("\nGenerating embeddings and calculating similarities...")
        try:
            results = utils.generate_and_compare(
                query=query,
                documents=documents,
                model_path=args.model,
                matryoshka_dim=args.matryoshka_dim,
            )

            print("\n===== Results =====")
            for i, res in enumerate(results):
                print(f"{i+1}. Similarity: {res['similarity']:.4f}")
                print(f"   Document: {res['document']}")
                print()

        except Exception as e:
            logger.error(f"Error in interactive mode: {str(e)}")
            print(f"Error: {str(e)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Embedding Model Evaluation")

    parser.add_argument(
        "--model", type=str, required=True, help="ID of the embedding model to evaluate"
    )
    parser.add_argument(
        "--samples", type=int, default=10, help="Number of test samples to use"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./evaluation/embeddings/data",
        help="Directory containing test data",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="basic_embedding_test.json",
        help="Name of the test data file",
    )
    parser.add_argument(
        "--output", type=str, help="Output file to save results (JSON format)"
    )
    parser.add_argument(
        "--matryoshka-dim",
        type=int,
        choices=[256, 512, 768],
        help="Matryoshka dimension for truncated embeddings",
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode(args)
    else:
        run_embedding_evaluation(args)


if __name__ == "__main__":
    main()
