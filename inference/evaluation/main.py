# main.py
import argparse
import logging
import statistics
from typing import List

from evaluation.suite import AcademicBenchmarkSuite


def main():
    """Main function to run academic benchmarks."""
    parser = argparse.ArgumentParser(description="Academic LLM Benchmark Suite")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen3-30b-a3b-q4-k-m"],
        help="Model IDs to benchmark (space-separated)",
    )
    parser.add_argument(
        "--output", type=str, help="Output file path for results (JSON format)"
    )
    parser.add_argument("--csv", action="store_true", help="Also generate CSV summary")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=[
            "mmlu",
            "gpqa_diamond",
            "math_500",
            "ifeval",
            "humaneval",
            "livebench",
            "all",
        ],
        default=["all"],
        help="Specific benchmarks to run (default: all)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples per benchmark (default: 50)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./benchmark_data",
        help="Directory for benchmark data (default: ./benchmark_data)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace dataset path in format 'space/dataset-name'",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    print("Academic LLM Benchmark Suite")
    print(f"Models to benchmark: {args.models}")
    print(f"Benchmarks to run: {args.benchmarks}")
    print(f"Samples per benchmark: {args.samples}")

    try:
        # Initialize benchmark suite
        benchmark_suite = AcademicBenchmarkSuite(data_dir=args.data_dir)

        # Run benchmarks
        all_results = benchmark_suite.run_full_benchmark_suite(
            model_ids=args.models,
            output_file=args.output,
            benchmarks_to_run=args.benchmarks,
            num_samples=args.samples,
            dataset_path=args.dataset,
        )

        # Save CSV if requested
        if args.csv and args.output:
            benchmark_suite.save_results_csv(all_results, args.output)

        logger.info("Academic benchmark suite completed successfully")

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        logger.info("Benchmark interrupted by user")

    except Exception as e:
        logger.error(f"Academic benchmark failed with error: {str(e)}", exc_info=True)
        print(f"Academic benchmark failed: {str(e)}")


if __name__ == "__main__":
    main()

# ===== Package Init Files =====

# benchmarks/__init__.
