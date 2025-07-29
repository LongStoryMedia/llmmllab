#!/usr/bin/env python3
"""
Direct runner script for model evaluations.

This script provides a simplified way to run the evaluation module directly.
"""

import argparse
import os
import subprocess
import json
import sys
from datetime import datetime


def run_evaluation(model_id="qwen3-30b-a3b-q4-k-m",
                   task="text-gen",
                   dataset="./evaluations/datasets/samples/text_generation_test.json",
                   output_dir="./results",
                   verbose=True):
    """Run the model evaluation with the given parameters."""

    print(f"Running evaluation for model: {model_id}")
    print(f"Task: {task}")
    print(f"Dataset: {dataset}")
    print(f"Output directory: {output_dir}")
    print("---")

    # Build command
    cmd = [
        "python", "-m", "evaluations.run_model_eval",
        "--model-id", model_id,
        "--task", task,
        "--dataset", dataset,
        "--output-dir", output_dir
    ]

    if verbose:
        cmd.append("--verbose")

    # Print the command
    print(f"Running command: {' '.join(cmd)}")

    # Run the command
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {str(e)}")
        print(e.stdout)
        print("STDERR:", e.stderr)
        return False

    # Check results
    model_output_dir = os.path.join(output_dir, model_id)
    main_results_pattern = f"{model_id}_{task}_results_*.json"

    print(f"\nResults should be in: {model_output_dir}/ and {output_dir}/")

    # List output directory
    if os.path.exists(output_dir):
        print(f"\nOutput directory contents: {output_dir}")
        for item in os.listdir(output_dir):
            if os.path.isfile(os.path.join(output_dir, item)):
                size = os.path.getsize(os.path.join(output_dir, item)) / 1024
                print(f"  {item} ({size:.1f} KB)")
            else:
                print(f"  {item}/ (directory)")

    # List model-specific directory
    if os.path.exists(model_output_dir):
        print(f"\nModel output directory contents: {model_output_dir}")
        results_files = []

        for item in os.listdir(model_output_dir):
            if os.path.isfile(os.path.join(model_output_dir, item)):
                size = os.path.getsize(os.path.join(
                    model_output_dir, item)) / 1024
                print(f"  {item} ({size:.1f} KB)")
                if f"{task}_results_" in item:
                    results_files.append(item)
            else:
                print(f"  {item}/ (directory)")

        # Show latest results file content
        if results_files:
            latest_file = sorted(results_files)[-1]
            results_path = os.path.join(model_output_dir, latest_file)

            print(f"\nLatest results file: {latest_file}")
            try:
                with open(results_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Print summary info
                print("\nResults summary:")
                print(f"  Model: {data.get('model_id')}")
                print(f"  Dataset: {data.get('dataset')}")
                print(f"  Timestamp: {data.get('timestamp')}")
                print(
                    f"  Average generation time: {data.get('avg_generation_time', 0):.2f}s")

                # Print metrics if available
                if 'bleu_scores' in data:
                    bleu = data['bleu_scores']
                    print(f"  BLEU score: {bleu.get('bleu', 0):.4f}")

                if 'rouge_scores' in data:
                    rouge = data['rouge_scores']
                    print(
                        f"  ROUGE-L F1: {rouge.get('rouge-l', {}).get('f', 0):.4f}")

            except Exception as e:
                print(f"Error reading results file: {str(e)}")

    return True


def main():
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Run model evaluation directly")
    parser.add_argument("--model-id", "-m", default="qwen3-30b-a3b-q4-k-m",
                        help="Model ID to evaluate")
    parser.add_argument("--task", "-t", default="text-gen",
                        choices=["text-gen", "multiple_choice", "mc", "qa"],
                        help="Type of evaluation to run")
    parser.add_argument("--dataset", "-d",
                        default="./evaluations/datasets/samples/text_generation_test.json",
                        help="Path to dataset JSON file")
    parser.add_argument("--output-dir", "-o", default="./results",
                        help="Directory to save results")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")

    args = parser.parse_args()

    success = run_evaluation(
        model_id=args.model_id,
        task=args.task,
        dataset=args.dataset,
        output_dir=args.output_dir,
        verbose=args.verbose
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
