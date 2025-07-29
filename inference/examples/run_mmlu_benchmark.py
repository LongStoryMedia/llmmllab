#!/usr/bin/env python3
"""
Example script to run the MMLU benchmark using the Hugging Face dataset.
"""
import sys
import os
import argparse
import logging

# Add the parent directory to the path so we can import the benchmark
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.academic.mmlu import MMLUBenchmark


def main():
    parser = argparse.ArgumentParser(
        description="Run MMLU benchmark with Hugging Face dataset"
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="Model ID to evaluate"
    )
    parser.add_argument(
        "--num-samples", type=int, default=20, help="Number of questions to evaluate"
    )
    parser.add_argument(
        "--use-huggingface", action="store_true", help="Use Hugging Face dataset"
    )
    parser.add_argument(
        "--subject",
        type=str,
        action="append",
        help="Filter by specific subjects (can be used multiple times)",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize benchmark
    benchmark = MMLUBenchmark()

    # Run benchmark with specified parameters
    result = benchmark.run(
        model_id=args.model_id,
        num_samples=args.num_samples,
        use_huggingface=args.use_huggingface,
        subjects=args.subject,
    )

    # Print summary
    print("\n--- MMLU Benchmark Results ---")
    print(f"Model: {args.model_id}")
    print(
        f"Score: {result.score:.3f} ({result.correct_answers}/{result.total_questions})"
    )

    # Print subject-wise scores if available
    if "subject_accuracy" in result.metadata:
        print("\nSubject-wise scores:")
        for subject, score in sorted(
            result.metadata["subject_accuracy"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(f"  {subject}: {score:.3f}")


if __name__ == "__main__":
    main()
