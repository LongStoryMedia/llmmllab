#!/usr/bin/env python
"""
Tool to compare the similarity between two pieces of text using Nomic Embed Text v2.
"""

import argparse
import logging
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to make imports work
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluation.embeddings.extractor import EmbeddingExtractor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def main():
    """Main function to compare text similarity."""
    parser = argparse.ArgumentParser(
        description="Compare text similarity using embeddings"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nomic-embed-text-v2",
        help="Model ID for the embedding model",
    )
    parser.add_argument("--text1", type=str, help="First text to compare")
    parser.add_argument("--text2", type=str, help="Second text to compare")
    parser.add_argument(
        "--is-query1",
        action="store_true",
        help="Treat text1 as a query (adds search_query: prefix)",
    )
    parser.add_argument(
        "--is-query2",
        action="store_true",
        help="Treat text2 as a query (adds search_query: prefix)",
    )
    parser.add_argument(
        "--file1", type=str, help="File containing first text (alternative to --text1)"
    )
    parser.add_argument(
        "--file2", type=str, help="File containing second text (alternative to --text2)"
    )
    parser.add_argument(
        "--matryoshka-dim",
        type=int,
        choices=[256, 512, 768],
        help="Matryoshka dimension for truncated embeddings",
    )

    args = parser.parse_args()

    # Read from files if specified
    text1 = args.text1
    text2 = args.text2

    if args.file1:
        with open(args.file1, "r", encoding="utf-8") as f:
            text1 = f.read()

    if args.file2:
        with open(args.file2, "r", encoding="utf-8") as f:
            text2 = f.read()

    if not text1 or not text2:
        print(
            "Error: Both text1 and text2 must be provided, either directly or through files"
        )
        sys.exit(1)

    # Initialize extractor
    extractor = EmbeddingExtractor()

    # Generate embeddings
    print("Generating embeddings...")
    try:
        embedding1 = extractor.generate_embedding(
            text=text1,
            model_id=args.model,
            is_query=args.is_query1,
            matryoshka_dim=args.matryoshka_dim,
        )

        embedding2 = extractor.generate_embedding(
            text=text2,
            model_id=args.model,
            is_query=args.is_query2,
            matryoshka_dim=args.matryoshka_dim,
        )

        # Calculate similarity
        similarity = cosine_similarity(embedding1, embedding2)

        print("\n===== Similarity Results =====")
        print(f"Cosine Similarity: {similarity:.6f}")
        print("\nInterpretation:")
        if similarity > 0.9:
            print("Very high similarity - texts are nearly identical in meaning")
        elif similarity > 0.8:
            print("High similarity - texts are very closely related")
        elif similarity > 0.6:
            print("Moderate similarity - texts are related")
        elif similarity > 0.4:
            print("Low similarity - texts have some connection")
        else:
            print("Very low similarity - texts are mostly unrelated")

    except (ValueError, ImportError, RuntimeError) as e:
        logger.error(f"Error comparing texts: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
