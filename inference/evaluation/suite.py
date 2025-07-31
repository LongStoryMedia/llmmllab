import datetime
import statistics
import json
import csv
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Store benchmark classes and availability status
benchmarks_available = {}
benchmark_classes = {}


# Define a function to safely import evaluation
def safe_import_benchmark(module_name: str, class_name: str, benchmark_key: str):
    """Safely import a benchmark class and store it if available."""
    try:
        module = importlib.import_module(
            f".academic.{module_name}", package="evaluation"
        )
        benchmark_class = getattr(module, class_name)
        benchmarks_available[benchmark_key] = True
        benchmark_classes[benchmark_key] = benchmark_class
    except (ImportError, AttributeError):
        benchmarks_available[benchmark_key] = False
        benchmark_classes[benchmark_key] = None


# Import all benchmarks
safe_import_benchmark("humaneval", "HumanEvalBenchmark", "humaneval")
safe_import_benchmark("mmlu", "MMLUBenchmark", "mmlu")
safe_import_benchmark("gpqa_diamond", "GPQADiamondBenchmark", "gpqa_diamond")
safe_import_benchmark("math_500", "Math500Benchmark", "math_500")
safe_import_benchmark("ifeval", "IFEvalBenchmark", "ifeval")
safe_import_benchmark("livebench", "LiveBenchmark", "livebench")


class AcademicBenchmarkSuite:
    """Main suite for running academic benchmarks."""

    def __init__(self, data_dir: str = "./benchmark_data"):
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Initialize available benchmarks
        self.benchmarks = {}

        # Create instances of available benchmarks
        for key, is_available in benchmarks_available.items():
            if (
                is_available
                and key in benchmark_classes
                and benchmark_classes[key] is not None
            ):
                try:
                    self.benchmarks[key] = benchmark_classes[key]()
                except Exception as e:
                    self.logger.warning(
                        f"Failed to initialize {key} benchmark: {str(e)}"
                    )

        if not self.benchmarks:
            self.logger.warning("No benchmark modules were successfully imported.")
        else:
            self.logger.info(
                f"Successfully loaded benchmarks: {list(self.benchmarks.keys())}"
            )

        self.setup_benchmark_data()

    def setup_benchmark_data(self):
        """Setup benchmark data directories."""
        print("Setting up benchmark datasets...")

        for benchmark_name in self.benchmarks.keys():
            (self.data_dir / benchmark_name).mkdir(exist_ok=True)

        self.logger.info("Benchmark data directories created")

    def run_full_benchmark_suite(
        self,
        model_ids: List[str],
        output_file: Optional[str] = None,
        benchmarks_to_run: Optional[List[str]] = None,
        num_samples: int = 50,
        dataset_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run all or specified academic benchmarks on specified models."""
        if benchmarks_to_run is None or "all" in benchmarks_to_run:
            benchmarks_to_run = list(self.benchmarks.keys())

        all_results = {}

        for model_id in model_ids:
            print(f"\n{'='*80}")
            print(f"RUNNING ACADEMIC BENCHMARKS: {model_id}")
            print(f"{'='*80}")

            model_results = {
                "model_id": model_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "benchmarks": {},
            }

            try:
                for benchmark_name in benchmarks_to_run:
                    if benchmark_name in self.benchmarks:
                        print(f"\n🧠 Running {benchmark_name.upper()}...")

                        # Adjust sample sizes for different benchmarks
                        if benchmark_name == "humaneval":
                            samples = num_samples
                        elif benchmark_name in ["gpqa_diamond", "math_500", "ifeval"]:
                            samples = num_samples
                        elif benchmark_name == "livebench":
                            samples = num_samples
                        else:
                            samples = num_samples

                        if samples < 1:
                            self.logger.warning(
                                f"Adjusted sample size for {benchmark_name} is less than 1, setting to 1."
                            )
                            samples = 1

                        # Pass dataset_path to the benchmark run method
                        result = self.benchmarks[benchmark_name].run(
                            model_id=model_id,
                            num_samples=samples,
                            dataset_path=dataset_path,
                        )
                        model_results["benchmarks"][benchmark_name] = {
                            "score": result.score,
                            "total_questions": result.total_questions,
                            "correct_answers": result.correct_answers,
                            "detailed_results": result.detailed_results,
                            "metadata": result.metadata,
                        }

                # Calculate average score
                scores = []
                for benchmark_name, result in model_results["benchmarks"].items():
                    if isinstance(result, dict) and "score" in result:
                        scores.append(result["score"])

                model_results["average_score"] = (
                    statistics.mean(scores) if scores else 0.0
                )
                all_results[model_id] = model_results

            except Exception as e:
                self.logger.error(f"Error benchmarking {model_id}: {str(e)}")
                model_results["error"] = str(e)
                all_results[model_id] = model_results
                raise e

        if output_file:
            self.save_results(all_results, output_file)

        self.print_academic_summary(all_results)
        return all_results

    def save_results(self, results: Dict, output_file: str):
        """Save academic benchmark results to JSON file."""
        output_dir = Path(output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nAcademic benchmark results saved to: {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")

    def save_results_csv(self, results: Dict, output_file: str):
        """Save academic benchmark results to CSV file."""
        try:
            csv_file = output_file.replace(".json", "_academic.csv")
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)

                # Write header
                writer.writerow(
                    [
                        "Model ID",
                        "MMLU",
                        "GPQA-Diamond",
                        "MATH-500",
                        "IFEVAL",
                        "HumanEval",
                        "LiveBench",
                        "Average Score",
                    ]
                )

                # Write data for each model
                for model_id, model_results in results.items():
                    if "error" not in model_results:
                        benchmarks = model_results.get("benchmarks", {})
                        writer.writerow(
                            [
                                model_id,
                                benchmarks.get("mmlu", {}).get("score", 0),
                                benchmarks.get("gpqa_diamond", {}).get("score", 0),
                                benchmarks.get("math_500", {}).get("score", 0),
                                benchmarks.get("ifeval", {}).get("score", 0),
                                benchmarks.get("humaneval", {}).get("score", 0),
                                benchmarks.get("livebench", {}).get("score", 0),
                                model_results.get("average_score", 0),
                            ]
                        )

            print(f"Academic benchmark CSV saved to: {csv_file}")
        except Exception as e:
            self.logger.error(f"Error saving CSV: {str(e)}")

    def print_academic_summary(self, results: Dict):
        """Print a summary of academic benchmark results."""
        print(f"\n{'='*80}")
        print("ACADEMIC BENCHMARK RESULTS SUMMARY")
        print(f"{'='*80}")

        # Create a summary table
        print(
            f"{'Model':<25} {'MMLU':<8} {'GPQA':<8} {'MATH':<8} {'IFEVAL':<8} {'HumanEval':<10} {'LiveBench':<10} {'Average':<8}"
        )
        print("-" * 90)

        for model_id, model_results in results.items():
            if "error" in model_results:
                print(f"{model_id:<25} ERROR: {model_results['error']}")
                continue

            benchmarks = model_results.get("benchmarks", {})

            mmlu_score = benchmarks.get("mmlu", {}).get("score", 0)
            gpqa_score = benchmarks.get("gpqa_diamond", {}).get("score", 0)
            math_score = benchmarks.get("math_500", {}).get("score", 0)
            ifeval_score = benchmarks.get("ifeval", {}).get("score", 0)
            humaneval_score = benchmarks.get("humaneval", {}).get("score", 0)
            livebench_score = benchmarks.get("livebench", {}).get("score", 0)
            avg_score = model_results.get("average_score", 0)

            print(
                f"{model_id:<25} {mmlu_score:.3f}    {gpqa_score:.3f}    {math_score:.3f}    {ifeval_score:.3f}    {humaneval_score:.3f}      {livebench_score:.3f}      {avg_score:.3f}"
            )

        print(f"\n{'='*90}")
        print("BENCHMARK DESCRIPTIONS:")
        print("- MMLU: Massive Multitask Language Understanding (57 subjects)")
        print("- GPQA: Graduate-level physics, chemistry, biology questions")
        print("- MATH: Mathematical problem solving across multiple domains")
        print("- IFEVAL: Instruction following evaluation")
        print("- HumanEval: Code generation and functional correctness")
        print("- LiveBench: Contamination-free evaluation with fresh questions")
        print("- Average: Mean score across all benchmarks")
