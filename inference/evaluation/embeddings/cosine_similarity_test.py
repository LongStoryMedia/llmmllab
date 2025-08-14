"""
Simple script to compare similarity between two texts using the Nomic Embed Text model.

This script demonstrates how to use the embedding evaluation framework to compare
the semantic similarity between two texts.
"""

import argparse
import logging
import os
import sys

# Add the parent directory to the path to allow importing from sibling packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from evaluation.embeddings.adapter import EmbeddingModelAdapter
from evaluation.embeddings.utils import EmbeddingUtils


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def read_text_from_file(file_path):
    """Read text from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def main():
    """Compare similarity between two texts."""
    parser = argparse.ArgumentParser(description="Compare similarity between two texts")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the Nomic Embed Text GGUF model file",
    )
    parser.add_argument(
        "--text1", type=str, help="First text to compare (or use --file1)"
    )
    parser.add_argument(
        "--text2", type=str, help="Second text to compare (or use --file2)"
    )
    parser.add_argument(
        "--file1", type=str, help="File containing first text (alternative to --text1)"
    )
    parser.add_argument(
        "--file2", type=str, help="File containing second text (alternative to --text2)"
    )
    parser.add_argument(
        "--is_query1",
        action="store_true",
        help="Treat text1 as a query (adds search_query: prefix)",
    )
    parser.add_argument(
        "--is_query2",
        action="store_true",
        help="Treat text2 as a query (adds search_query: prefix)",
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
    logger = logging.getLogger(__name__)

    # Get text input
    text1 = args.text1
    text2 = args.text2

    if args.file1:
        text1 = read_text_from_file(args.file1)
    if args.file2:
        text2 = read_text_from_file(args.file2)

    if not text1 or not text2:
        logger.error(
            "Both text1 and text2 must be provided (via --text or --file options)"
        )
        sys.exit(1)

    # Create adapter and utils
    adapter = EmbeddingModelAdapter()
    utils = EmbeddingUtils()

    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = adapter.generate_embeddings(
        texts=[text1, text2],
        model_path=args.model_path,
        is_query=[args.is_query1, args.is_query2],
        matryoshka_dim=args.matryoshka_dim,
    )

    # Calculate similarity
    similarity = utils.cosine_similarity(embeddings[0], embeddings[1])

    # Print results
    logger.info("\nComparison Results:")
    logger.info(f"Text 1 ({len(text1)} chars): {text1[:100]}...")
    if args.is_query1:
        logger.info("  (Processed as search query)")

    logger.info(f"Text 2 ({len(text2)} chars): {text2[:100]}...")
    if args.is_query2:
        logger.info("  (Processed as search query)")

    logger.info(f"\nSimilarity Score: {similarity:.4f}")

    # Interpret similarity
    if similarity > 0.8:
        interpretation = "Very High - Texts have very similar meaning"
    elif similarity > 0.65:
        interpretation = "High - Texts are closely related"
    elif similarity > 0.5:
        interpretation = "Moderate - Texts are somewhat related"
    elif similarity > 0.3:
        interpretation = "Low - Texts have minimal relation"
    else:
        interpretation = "Very Low - Texts are unrelated"

    logger.info(f"Interpretation: {interpretation}")

    if args.matryoshka_dim:
        logger.info(
            f"Note: Used {args.matryoshka_dim}-dimensional Matryoshka embeddings"
        )


if __name__ == "__main__":
    main()
