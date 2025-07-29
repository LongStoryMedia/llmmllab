#!/usr/bin/env python3
"""
Benchmark Results Visualizer

This script visualizes benchmark results from the LLM benchmarking framework.
It creates both time series graphs (showing performance over time) and bar charts
(showing current/latest results).

Usage:
    python visualize_results.py results.json
    python visualize_results.py results1.json results2.json results3.json --output charts/
"""

import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


class BenchmarkVisualizer:
    """Visualizes benchmark results from JSON files."""

    def __init__(self, output_dir: str = "./bencmark_data/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up plotting style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # Benchmark display names and colors
        self.benchmark_names = {
            "mmlu": "MMLU",
            "gpqa_diamond": "GPQA-Diamond",
            "math_500": "MATH-500",
            "ifeval": "IFEval",
            "humaneval": "HumanEval",
            "livebench": "LiveBench",
        }

        self.benchmark_colors = {
            "mmlu": "#1f77b4",
            "gpqa_diamond": "#ff7f0e",
            "math_500": "#2ca02c",
            "ifeval": "#d62728",
            "humaneval": "#9467bd",
            "livebench": "#8c564b",
        }

    def load_results(self, result_files: List[str]) -> List[Dict[str, Any]]:
        """Load benchmark results from JSON files."""
        all_results = []

        for file_path in result_files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                # Handle different JSON structures
                if isinstance(data, dict):
                    # If it's a single run result, wrap in list
                    if "model_id" in data or any(
                        key in data for key in self.benchmark_names.keys()
                    ):
                        all_results.append({"file": file_path, "data": data})
                    else:
                        # If it contains multiple models
                        for model_id, model_data in data.items():
                            all_results.append(
                                {
                                    "file": file_path,
                                    "model_id": model_id,
                                    "data": model_data,
                                }
                            )
                elif isinstance(data, list):
                    for item in data:
                        all_results.append({"file": file_path, "data": item})

                print(f"Loaded results from: {file_path}")

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        return all_results

    def extract_benchmark_data(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract benchmark data into a pandas DataFrame."""
        rows = []

        for result in results:
            data = result["data"]
            file_name = Path(result["file"]).stem

            # Extract model ID
            model_id = result.get("model_id") or data.get("model_id", "unknown_model")

            # Extract timestamp
            timestamp_str = data.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = pd.to_datetime(timestamp_str)
                except:
                    timestamp = pd.to_datetime("now")
            else:
                # Use file modification time as fallback
                try:
                    timestamp = pd.to_datetime(
                        os.path.getmtime(result["file"]), unit="s"
                    )
                except:
                    timestamp = pd.to_datetime("now")

            # Extract benchmark scores
            benchmarks = data.get("benchmarks", {})
            average_score = data.get("average_score", 0)

            row = {
                "model_id": model_id,
                "timestamp": timestamp,
                "file": file_name,
                "average_score": average_score,
            }

            # Add individual benchmark scores
            for bench_key, bench_name in self.benchmark_names.items():
                if bench_key in benchmarks:
                    bench_data = benchmarks[bench_key]
                    if isinstance(bench_data, dict):
                        row[bench_key] = bench_data.get("score", 0)
                    else:
                        row[bench_key] = bench_data
                else:
                    row[bench_key] = 0

            rows.append(row)

        return pd.DataFrame(rows)

    def create_time_series_plot(self, df: pd.DataFrame) -> Figure:
        """Create time series plot showing benchmark performance over time."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        fig.suptitle("Benchmark Performance Over Time", fontsize=16, fontweight="bold")

        # Plot each benchmark
        for i, (bench_key, bench_name) in enumerate(self.benchmark_names.items()):
            ax = axes[i]

            # Group by model and plot each model's performance over time
            for j, model_id in enumerate(df["model_id"].unique()):
                model_data = (
                    df[df["model_id"] == model_id]
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )

                if len(model_data) > 0 and bench_key in model_data.columns:
                    # Use evenly spaced integers for x-axis
                    x_values = range(len(model_data))
                    ax.plot(
                        x_values,
                        model_data[bench_key],
                        marker="o",
                        linewidth=2,
                        markersize=6,
                        label=model_id,
                    )

            ax.set_title(f"{bench_name} Performance", fontweight="bold")
            ax.set_xlabel("Run Number")
            ax.set_ylabel("Score")
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Set integer ticks on x-axis
            if len(df) > 0:
                max_runs = max(
                    len(df[df["model_id"] == model]["timestamp"].unique())
                    for model in df["model_id"].unique()
                )
                ax.set_xticks(range(max_runs))

        # Remove empty subplot if we have fewer than 6 benchmarks
        if len(self.benchmark_names) < 6:
            axes[-1].remove()

        plt.tight_layout()
        return fig

    def create_average_time_series(self, df: pd.DataFrame) -> Figure:
        """Create time series plot for average scores across all benchmarks."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot average score over time for each model
        for model_id in df["model_id"].unique():
            model_data = (
                df[df["model_id"] == model_id]
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

            if len(model_data) > 0:
                # Use evenly spaced integers for x-axis
                x_values = range(len(model_data))
                ax.plot(
                    x_values,
                    model_data["average_score"],
                    marker="o",
                    linewidth=3,
                    markersize=8,
                    label=model_id,
                )

        ax.set_title(
            "Average Benchmark Performance Over Time", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Run Number")
        ax.set_ylabel("Average Score")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set integer ticks on x-axis
        if len(df) > 0:
            max_runs = max(
                len(df[df["model_id"] == model]["timestamp"].unique())
                for model in df["model_id"].unique()
            )
            ax.set_xticks(range(max_runs))

        plt.tight_layout()
        return fig

    def create_current_results_bar_chart(self, df: pd.DataFrame) -> Figure:
        """Create bar chart showing the most recent results for each model."""
        # Get the most recent results for each model
        latest_results = df.loc[df.groupby("model_id")["timestamp"].idxmax()]

        if len(latest_results) == 0:
            print("No data available for bar chart")
            return None

        # Create subplots for individual benchmarks and average
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        fig.suptitle("Latest Benchmark Results", fontsize=16, fontweight="bold")

        # Plot individual benchmarks
        for i, (bench_key, bench_name) in enumerate(self.benchmark_names.items()):
            ax = axes[i]

            if bench_key in latest_results.columns:
                bars = ax.bar(
                    latest_results["model_id"],
                    latest_results[bench_key],
                    color=self.benchmark_colors.get(bench_key, "#000000"),
                    alpha=0.7,
                )

                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.01,
                        f"{height:.3f}",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

            ax.set_title(bench_name, fontweight="bold")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1.1)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Plot average scores
        ax = axes[6]
        bars = ax.bar(
            latest_results["model_id"],
            latest_results["average_score"],
            color="#34495e",
            alpha=0.8,
        )

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_title("Average Score", fontweight="bold")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.1)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Remove empty subplot
        axes[7].remove()

        plt.tight_layout()
        return fig

    def create_model_comparison_charts(self, df: pd.DataFrame) -> List[Figure]:
        """Create comprehensive model comparison charts."""
        figures = []

        # Get latest results for comparison
        latest_results = df.loc[df.groupby("model_id")["timestamp"].idxmax()]

        if len(latest_results) < 2:
            print("Need at least 2 models for comparison charts")
            return figures

        # 1. Side-by-side comparison for all benchmarks
        fig1 = self.create_side_by_side_comparison(latest_results)
        if fig1:
            figures.append(("model_comparison_all", fig1))

        # 2. Radar chart comparison
        fig2 = self.create_radar_comparison(latest_results)
        if fig2:
            figures.append(("model_comparison_radar", fig2))

        # 3. Performance gap analysis
        fig3 = self.create_performance_gap_analysis(latest_results)
        if fig3:
            figures.append(("model_performance_gaps", fig3))

        # 4. Win/Loss comparison matrix
        fig4 = self.create_win_loss_matrix(latest_results)
        if fig4:
            figures.append(("model_win_loss_matrix", fig4))

        return figures

    def create_side_by_side_comparison(self, latest_results: pd.DataFrame) -> Figure:
        """Create side-by-side bar comparison of all models."""
        fig, ax = plt.subplots(figsize=(14, 8))

        models = latest_results["model_id"].tolist()
        benchmarks = list(self.benchmark_names.keys())

        x = range(len(benchmarks))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            model_data = latest_results[latest_results["model_id"] == model]
            scores = [
                model_data[bench].iloc[0] if bench in model_data.columns else 0
                for bench in benchmarks
            ]

            offset = (i - len(models) / 2 + 0.5) * width
            bars = ax.bar(
                [pos + offset for pos in x], scores, width, label=model, alpha=0.8
            )

            # Add value labels on bars
            for bar, score in zip(bars, scores):
                if score > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + 0.01,
                        f"{score:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

        ax.set_title(
            "Model Performance Comparison (Latest Results)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Benchmarks")
        ax.set_ylabel("Score")
        ax.set_xticks(x)
        ax.set_xticklabels([self.benchmark_names[b] for b in benchmarks], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        return fig

    def create_radar_comparison(self, latest_results: pd.DataFrame) -> Figure:
        """Create radar chart comparing models across all benchmarks."""
        try:
            import numpy as np

            models = latest_results["model_id"].tolist()
            benchmarks = list(self.benchmark_names.keys())

            # Number of variables
            num_vars = len(benchmarks)

            # Compute angle for each axis
            angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
            angles += angles[:1]  # Complete the circle

            fig, ax = plt.subplots(
                figsize=(10, 10), subplot_kw=dict(projection="polar")
            )

            # Plot each model
            colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

            for i, model in enumerate(models):
                model_data = latest_results[latest_results["model_id"] == model]
                values = [
                    model_data[bench].iloc[0] if bench in model_data.columns else 0
                    for bench in benchmarks
                ]
                values += values[:1]  # Complete the circle

                ax.plot(angles, values, "o-", linewidth=2, label=model, color=colors[i])
                ax.fill(angles, values, alpha=0.25, color=colors[i])

            # Add labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([self.benchmark_names[b] for b in benchmarks])
            ax.set_ylim(0, 1)
            ax.set_title(
                "Model Performance Radar Chart", size=16, fontweight="bold", pad=20
            )
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)

            plt.tight_layout()
            return fig

        except ImportError:
            print("Numpy required for radar chart - skipping")
            return None

    def create_performance_gap_analysis(self, latest_results: pd.DataFrame) -> Figure:
        """Create chart showing performance gaps between models."""
        if len(latest_results) < 2:
            return None

        fig, ax = plt.subplots(figsize=(12, 8))

        benchmarks = list(self.benchmark_names.keys())
        models = latest_results["model_id"].tolist()

        # Calculate gaps from the best performing model for each benchmark
        gaps = []
        benchmark_labels = []

        for bench in benchmarks:
            if bench in latest_results.columns:
                scores = latest_results[bench].values
                max_score = max(scores)

                for i, (model, score) in enumerate(zip(models, scores)):
                    gap = max_score - score
                    gaps.append(gap)
                    benchmark_labels.append(f"{model}\n{self.benchmark_names[bench]}")

        # Create grouped bar chart
        y_pos = range(len(gaps))
        bars = ax.barh(y_pos, gaps, alpha=0.7)

        # Color bars by gap size
        for i, (bar, gap) in enumerate(zip(bars, gaps)):
            if gap == 0:
                bar.set_color("green")
            elif gap < 0.1:
                bar.set_color("orange")
            else:
                bar.set_color("red")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(benchmark_labels, fontsize=8)
        ax.set_xlabel("Performance Gap from Best Model")
        ax.set_title("Performance Gap Analysis", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        return fig

    def create_win_loss_matrix(self, latest_results: pd.DataFrame) -> Figure:
        """Create matrix showing which model wins on each benchmark."""
        if len(latest_results) < 2:
            return None

        models = latest_results["model_id"].tolist()
        benchmarks = list(self.benchmark_names.keys())

        # Create win matrix
        win_matrix = []

        for model in models:
            model_wins = []
            model_data = latest_results[latest_results["model_id"] == model]

            for bench in benchmarks:
                if bench in latest_results.columns:
                    all_scores = latest_results[bench].values
                    model_score = model_data[bench].iloc[0]
                    # Count how many models this model beats on this benchmark
                    wins = sum(1 for score in all_scores if model_score > score)
                    model_wins.append(wins)
                else:
                    model_wins.append(0)

            win_matrix.append(model_wins)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create heatmap
        im = ax.imshow(win_matrix, cmap="RdYlGn", aspect="auto")

        # Add text annotations
        for i in range(len(models)):
            for j in range(len(benchmarks)):
                text = ax.text(
                    j,
                    i,
                    win_matrix[i][j],
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                )

        ax.set_xticks(range(len(benchmarks)))
        ax.set_xticklabels([self.benchmark_names[b] for b in benchmarks], rotation=45)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.set_title(
            "Model Win Count Matrix\n(Number of models beaten on each benchmark)",
            fontweight="bold",
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Number of Models Beaten", rotation=270, labelpad=15)

        plt.tight_layout()
        return fig

    def create_comparison_heatmap(self, df: pd.DataFrame) -> Figure:
        """Create heatmap comparing latest results across models and benchmarks."""
        # Get the most recent results for each model
        latest_results = df.loc[df.groupby("model_id")["timestamp"].idxmax()]

        if len(latest_results) == 0:
            return None

        # Prepare data for heatmap
        benchmark_cols = list(self.benchmark_names.keys())
        heatmap_data = latest_results.set_index("model_id")[benchmark_cols]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create heatmap
        sns.heatmap(
            heatmap_data.T,
            annot=True,
            cmap="RdYlGn",
            center=0.5,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            fmt=".3f",
            ax=ax,
        )

        ax.set_title(
            "Benchmark Performance Comparison (Latest Results)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Models")
        ax.set_ylabel("Benchmarks")

        # Update y-axis labels with display names
        y_labels = [
            self.benchmark_names.get(label.get_text(), label.get_text())
            for label in ax.get_yticklabels()
        ]
        ax.set_yticklabels(y_labels, rotation=0)

        plt.tight_layout()
        return fig

    def generate_all_visualizations(self, result_files: List[str]):
        """Generate all visualizations and save them."""
        print("Loading benchmark results...")
        results = self.load_results(result_files)

        if not results:
            print("No valid results found!")
            return

        print("Processing data...")
        df = self.extract_benchmark_data(results)

        if df.empty:
            print("No benchmark data found!")
            return

        print(f"Found data for {len(df)} runs across {df['model_id'].nunique()} models")
        print("Models:", df["model_id"].unique().tolist())

        # Generate visualizations
        visualizations = [
            (
                "time_series_individual",
                "Individual Benchmark Time Series",
                self.create_time_series_plot,
            ),
            (
                "time_series_average",
                "Average Score Time Series",
                self.create_average_time_series,
            ),
            (
                "current_results_bars",
                "Latest Results Bar Chart",
                self.create_current_results_bar_chart,
            ),
            (
                "comparison_heatmap",
                "Performance Comparison Heatmap",
                self.create_comparison_heatmap,
            ),
        ]

        for filename, title, create_func in visualizations:
            try:
                print(f"Creating {title}...")
                fig = create_func(df)

                if fig is not None:
                    output_path = self.output_dir / f"{filename}.png"
                    fig.savefig(output_path, dpi=300, bbox_inches="tight")
                    print(f"Saved: {output_path}")
                    plt.close(fig)
                else:
                    print(f"Skipped {title} (no data)")

            except Exception as e:
                print(f"Error creating {title}: {e}")

        # Generate model comparison charts
        print("Creating model comparison charts...")
        comparison_figures = self.create_model_comparison_charts(df)

        for filename, fig in comparison_figures:
            try:
                output_path = self.output_dir / f"{filename}.png"
                fig.savefig(output_path, dpi=300, bbox_inches="tight")
                print(f"Saved: {output_path}")
                plt.close(fig)
            except Exception as e:
                print(f"Error saving {filename}: {e}")

        # Generate summary stats
        self.generate_summary_report(df)

    def generate_summary_report(self, df: pd.DataFrame):
        """Generate a text summary report."""
        report_path = self.output_dir / "summary_report.txt"

        with open(report_path, "w") as f:
            f.write("BENCHMARK RESULTS SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(
                f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Total runs analyzed: {len(df)}\n")
            f.write(f"Models evaluated: {df['model_id'].nunique()}\n")
            f.write(
                f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}\n\n"
            )

            # Latest results summary
            latest_results = df.loc[df.groupby("model_id")["timestamp"].idxmax()]

            f.write("LATEST RESULTS BY MODEL:\n")
            f.write("-" * 30 + "\n")

            for _, row in latest_results.iterrows():
                f.write(f"\nModel: {row['model_id']}\n")
                f.write(f"Date: {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Average Score: {row['average_score']:.3f}\n")

                f.write("Individual Benchmarks:\n")
                for bench_key, bench_name in self.benchmark_names.items():
                    if bench_key in row and pd.notna(row[bench_key]):
                        f.write(f"  {bench_name}: {row[bench_key]:.3f}\n")

            # Best performing models
            f.write(f"\nTOP PERFORMING MODELS (by average score):\n")
            f.write("-" * 40 + "\n")
            top_models = latest_results.nlargest(5, "average_score")

            for i, (_, row) in enumerate(top_models.iterrows(), 1):
                f.write(f"{i}. {row['model_id']}: {row['average_score']:.3f}\n")

        print(f"Summary report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize benchmark results from JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_results.py results.json
  python visualize_results.py results1.json results2.json results3.json
  python visualize_results.py "*.json" --output ./bencmark_data/charts/
  python visualize_results.py "./benchmark_data/*/model_name/*.json" --output ./bencmark_data/charts/
        """,
    )

    parser.add_argument(
        "result_files",
        nargs="+",
        help="JSON result files to visualize (glob patterns supported)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./bencmark_data/charts",
        help="Output directory for charts (default: ./bencmark_data/charts)",
    )

    args = parser.parse_args()

    # Validate input files
    valid_files = []
    # Expand glob patterns in result_files
    expanded_files = []
    for pattern in args.result_files:
        # Use glob to expand the pattern
        matched_files = glob.glob(pattern, recursive=True)
        if matched_files:
            expanded_files.extend(matched_files)
        else:
            print(f"Warning: No files found matching pattern: {pattern}")

    # Filter to valid files that exist
    for file_path in expanded_files:
        if os.path.exists(file_path):
            valid_files.append(file_path)
        else:
            print(f"Warning: File not found: {file_path}")

    if not valid_files:
        print("Error: No valid input files found!")
        return 1

    # Create visualizer and generate charts
    visualizer = BenchmarkVisualizer(output_dir=args.output)
    visualizer.generate_all_visualizations(valid_files)

    print(f"\nVisualization complete! Charts saved to: {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
