# Benchmark Results Visualizer

A comprehensive Python script for visualizing LLM benchmark results from JSON files. This tool creates multiple chart types including time series plots, comparison charts, and statistical summaries to help analyze model performance across different benchmarks.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command Line Arguments](#command-line-arguments)
- [Supported Input Formats](#supported-input-formats)
- [Chart Types Generated](#chart-types-generated)
- [Use Cases & Examples](#use-cases--examples)
- [Output Files](#output-files)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.7 or higher
- Required packages:

```bash
pip install matplotlib pandas seaborn numpy
```

### Quick Install

```bash
# Clone or download the script
wget https://your-repo/visualize_results.py

# Install dependencies
pip install matplotlib pandas seaborn numpy

# Make executable (optional)
chmod +x visualize_results.py
```

## Quick Start

```bash
# Basic usage - single file
python visualize_results.py results.json

# Multiple files
python visualize_results.py results1.json results2.json results3.json

# Custom output directory
python visualize_results.py *.json --output ./my_charts/
```

## Command Line Arguments

### Positional Arguments

#### `result_files` (required)
- **Type**: One or more file paths
- **Description**: JSON result files to visualize
- **Format**: Space-separated list of file paths
- **Supports**: Wildcards (`*.json`), relative paths, absolute paths

**Examples:**
```bash
# Single file
python visualize_results.py benchmark_results.json

# Multiple specific files
python visualize_results.py day1.json day2.json day3.json

# All JSON files in current directory
python visualize_results.py *.json

# Files from different directories
python visualize_results.py ./results/model1.json ./results/model2.json

# Mixed paths
python visualize_results.py today.json ./archive/yesterday.json /home/user/old_results.json
```

### Optional Arguments

#### `--output` / `-o`
- **Type**: String (directory path)
- **Default**: `./charts`
- **Description**: Output directory where all generated charts will be saved
- **Behavior**: Creates directory if it doesn't exist

**Examples:**
```bash
# Use default output directory (./charts)
python visualize_results.py results.json

# Custom output directory
python visualize_results.py results.json --output /home/user/benchmark_charts

# Relative path
python visualize_results.py results.json -o ../analysis/charts

# Create nested directories
python visualize_results.py results.json --output ./reports/2024/january/
```

#### `--help` / `-h`
- **Description**: Show help message and exit
- **Usage**: `python visualize_results.py --help`

## Supported Input Formats

The script supports multiple JSON structures from your benchmark suite:

### Format 1: Single Model Result
```json
{
  "model_id": "gpt-4",
  "timestamp": "2024-01-15T10:30:00",
  "benchmarks": {
    "mmlu": {"score": 0.847, "total_questions": 100, "correct_answers": 85},
    "humaneval": {"score": 0.672, "total_questions": 20, "correct_answers": 13},
    "math_500": {"score": 0.543, "total_questions": 50, "correct_answers": 27}
  },
  "average_score": 0.687
}
```

### Format 2: Multi-Model Results
```json
{
  "gpt-4": {
    "model_id": "gpt-4",
    "timestamp": "2024-01-15T10:30:00",
    "benchmarks": {...},
    "average_score": 0.687
  },
  "claude-3": {
    "model_id": "claude-3",
    "timestamp": "2024-01-15T11:45:00",
    "benchmarks": {...},
    "average_score": 0.692
  }
}
```

### Format 3: Array of Results
```json
[
  {
    "model_id": "gpt-4",
    "timestamp": "2024-01-15T10:30:00",
    "benchmarks": {...},
    "average_score": 0.687
  },
  {
    "model_id": "claude-3",
    "timestamp": "2024-01-15T11:45:00",
    "benchmarks": {...},
    "average_score": 0.692
  }
]
```

### Supported Benchmarks
- **MMLU**: Massive Multitask Language Understanding
- **GPQA-Diamond**: Graduate-level Physics, Chemistry, Biology Q&A
- **MATH-500**: Mathematical problem solving
- **IFEval**: Instruction Following Evaluation
- **HumanEval**: Code generation and functional correctness
- **LiveBench**: Contamination-free evaluation

## Chart Types Generated

### 1. Individual Benchmark Time Series (`time_series_individual.png`)
- **Purpose**: Track performance trends for each benchmark separately
- **X-axis**: Run number (chronologically ordered, evenly spaced integers)
- **Y-axis**: Score (0.0 to 1.0)
- **Features**: 
  - Separate subplot for each benchmark
  - Multiple model lines on each plot
  - Grid lines for easy reading
  - Legend showing all models

### 2. Average Score Time Series (`time_series_average.png`)
- **Purpose**: Track overall performance trends across all benchmarks
- **X-axis**: Run number (chronologically ordered)
- **Y-axis**: Average score across all benchmarks
- **Features**:
  - Single plot with all models
  - Thick lines with large markers
  - Clear legend

### 3. Latest Results Bar Chart (`current_results_bars.png`)
- **Purpose**: Compare most recent performance across models and benchmarks
- **Layout**: 2x4 grid of subplots
- **Features**:
  - Individual benchmark results + average score
  - Value labels on top of bars
  - Color-coded by benchmark
  - Model names on x-axis

### 4. Performance Comparison Heatmap (`comparison_heatmap.png`)
- **Purpose**: Matrix view of latest model performance
- **Format**: Models vs Benchmarks grid
- **Features**:
  - Color-coded performance (red=low, green=high)
  - Numerical values in each cell
  - Easy identification of strengths/weaknesses

### 5. Side-by-Side Model Comparison (`model_comparison_all.png`)
- **Purpose**: Direct comparison of all models on all benchmarks
- **Format**: Grouped bar chart
- **Features**:
  - Models grouped by benchmark
  - Value labels on bars
  - Legend for model identification
  - Grid lines for easy reading

### 6. Radar Chart Comparison (`model_comparison_radar.png`)
- **Purpose**: Multi-dimensional view of model capabilities
- **Format**: Circular/spider plot
- **Features**:
  - Each benchmark as a spoke
  - Filled areas showing model "shape"
  - Easy visualization of strengths/weaknesses
  - Multiple models overlaid

### 7. Performance Gap Analysis (`model_performance_gaps.png`)
- **Purpose**: Show how far behind each model is from the best performer
- **Format**: Horizontal bar chart
- **Features**:
  - Color-coded gaps (green=best, orange=close, red=far)
  - Grouped by model and benchmark
  - Shows exact gap values

### 8. Win/Loss Matrix (`model_win_loss_matrix.png`)
- **Purpose**: Competitive analysis showing which models beat others
- **Format**: Heatmap matrix
- **Features**:
  - Numbers show count of models beaten
  - Color-coded for quick assessment
  - Models vs Benchmarks layout

## Use Cases & Examples

### Use Case 1: Single Model Progress Tracking

**Scenario**: Track how your model improves over time during training/fine-tuning.

```bash
# Files from different training checkpoints
python visualize_results.py \
  model_checkpoint_1000.json \
  model_checkpoint_2000.json \
  model_checkpoint_3000.json \
  model_checkpoint_4000.json \
  --output ./training_progress/
```

**Generated Charts**:
- Time series showing improvement trends
- Latest results showing final performance
- No comparison charts (single model)

### Use Case 2: Model Competition Analysis

**Scenario**: Compare multiple different models to find the best performer.

```bash
# Compare different model architectures
python visualize_results.py \
  gpt4_results.json \
  claude3_results.json \
  llama2_results.json \
  gemini_results.json \
  --output ./model_comparison/
```

**Generated Charts**:
- All 8 chart types
- Comprehensive model comparison
- Competitive analysis charts
- Performance gap analysis

### Use Case 3: Longitudinal Study

**Scenario**: Track multiple models over an extended period.

```bash
# Weekly evaluation results over 3 months
python visualize_results.py \
  week_*.json \
  --output ./longitudinal_study/
```

**File Structure Example**:
```
week_01_gpt4.json
week_01_claude3.json
week_02_gpt4.json
week_02_claude3.json
...
week_12_gpt4.json
week_12_claude3.json
```

### Use Case 4: Benchmark-Specific Analysis

**Scenario**: Focus on specific benchmark performance across models.

```bash
# All models, focus on code generation
python visualize_results.py \
  humaneval_results_*.json \
  --output ./code_generation_analysis/
```

### Use Case 5: A/B Testing Results

**Scenario**: Compare two versions of the same model.

```bash
# Compare baseline vs experimental version
python visualize_results.py \
  baseline_model_v1.json \
  experimental_model_v2.json \
  --output ./ab_test_results/
```

### Use Case 6: Batch Processing Multiple Experiments

**Scenario**: Process all results from a research project.

```bash
# Process all experiments in a directory
python visualize_results.py \
  ./experiments/*/results.json \
  --output ./final_analysis/

# Alternative with find
find ./experiments -name "*.json" -exec python visualize_results.py {} + --output ./final_analysis/
```

### Use Case 7: Daily Monitoring

**Scenario**: Daily automated evaluation and visualization.

```bash
#!/bin/bash
# daily_monitoring.sh

DATE=$(date +%Y%m%d)
OUTPUT_DIR="./daily_reports/$DATE"

# Run benchmarks (your existing pipeline)
# ... benchmark execution code ...

# Generate visualizations
python visualize_results.py \
  ./daily_results/$DATE/*.json \
  --output "$OUTPUT_DIR"

echo "Daily report generated: $OUTPUT_DIR"
```

### Use Case 8: Research Paper Figures

**Scenario**: Generate publication-quality figures for research papers.

```bash
# High-quality figures for paper
python visualize_results.py \
  paper_experiment_*.json \
  --output ./paper_figures/

# Then manually select and edit the most relevant charts
```

### Use Case 9: Real-time Dashboard Data

**Scenario**: Generate data for a live dashboard.

```bash
# Generate latest comparison charts every hour
while true; do
  python visualize_results.py \
    ./live_results/*.json \
    --output ./dashboard_data/
  
  sleep 3600  # Wait 1 hour
done
```

### Use Case 10: Model Selection for Production

**Scenario**: Choose the best model for deployment.

```bash
# Compare candidate models for production
python visualize_results.py \
  candidate_model_*.json \
  --output ./production_selection/
```

## Output Files

When you run the script, it generates the following files in the output directory:

### Chart Files (PNG format, 300 DPI)
1. `time_series_individual.png` - Individual benchmark trends (18x12 inches)
2. `time_series_average.png` - Average score trends (12x6 inches)
3. `current_results_bars.png` - Latest results comparison (20x10 inches)
4. `comparison_heatmap.png` - Performance matrix (10x6 inches)
5. `model_comparison_all.png` - Side-by-side comparison (14x8 inches)
6. `model_comparison_radar.png` - Radar chart (10x10 inches)
7. `model_performance_gaps.png` - Gap analysis (12x8 inches)
8. `model_win_loss_matrix.png` - Win/loss matrix (10x6 inches)

### Report Files
9. `summary_report.txt` - Text summary with statistics

### Example Output Directory Structure
```
charts/
├── time_series_individual.png
├── time_series_average.png
├── current_results_bars.png
├── comparison_heatmap.png
├── model_comparison_all.png
├── model_comparison_radar.png
├── model_performance_gaps.png
├── model_win_loss_matrix.png
└── summary_report.txt
```

## Configuration

### Default Settings

```python
# Chart styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Output settings
DPI = 300  # High resolution for publications
FORMAT = 'PNG'  # Image format

# Default colors for benchmarks
benchmark_colors = {
    'mmlu': '#1f77b4',        # Blue
    'gpqa_diamond': '#ff7f0e', # Orange
    'math_500': '#2ca02c',     # Green
    'ifeval': '#d62728',       # Red
    'humaneval': '#9467bd',    # Purple
    'livebench': '#8c564b'     # Brown
}
```

### Customizing Colors

To customize colors, edit the `benchmark_colors` dictionary in the script:

```python
self.benchmark_colors = {
    'mmlu': '#your_color_here',
    'gpqa_diamond': '#another_color',
    # ... etc
}
```

### Customizing Chart Sizes

Chart sizes can be modified in each chart creation function:

```python
# Example: Make larger time series charts
fig, axes = plt.subplots(2, 3, figsize=(24, 16))  # Increased from (18, 12)
```

## Advanced Usage Examples

### Example 1: Filtering by Date Range

```bash
# Only process files from last week
find ./results -name "*.json" -mtime -7 -exec python visualize_results.py {} + --output ./recent_analysis/
```

### Example 2: Processing Large Numbers of Files

```bash
# Process files in batches to avoid command line length limits
ls results/*.json | xargs -n 20 python visualize_results.py --output ./batch_analysis/
```

### Example 3: Automated Report Generation

```bash
#!/bin/bash
# generate_weekly_report.sh

WEEK=$(date +%Y_week_%V)
mkdir -p "./reports/$WEEK"

# Generate charts
python visualize_results.py \
  ./weekly_results/*.json \
  --output "./reports/$WEEK/"

# Create summary
echo "Weekly Benchmark Report - $WEEK" > "./reports/$WEEK/README.md"
echo "Generated on: $(date)" >> "./reports/$WEEK/README.md"
echo "" >> "./reports/$WEEK/README.md"
echo "## Files Generated:" >> "./reports/$WEEK/README.md"
ls "./reports/$WEEK/"*.png >> "./reports/$WEEK/README.md"
```

### Example 4: Conditional Processing

```bash
#!/bin/bash
# Only generate charts if new results are available

LATEST_RESULT=$(find ./results -name "*.json" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
LATEST_CHART=$(find ./charts -name "*.png" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

if [[ "$LATEST_RESULT" -nt "$LATEST_CHART" ]]; then
    echo "New results found, generating charts..."
    python visualize_results.py ./results/*.json --output ./charts/
else
    echo "No new results, skipping chart generation"
fi
```

## Error Handling & Edge Cases

### Handling Missing Data

The script gracefully handles:
- Missing benchmark scores (displays as 0)
- Missing timestamps (uses file modification time)
- Empty or malformed JSON files (skips with warning)
- Missing model IDs (labels as "unknown_model")

### File Validation

```bash
# Check if files exist before processing
for file in model1.json model2.json model3.json; do
    if [[ ! -f "$file" ]]; then
        echo "Warning: $file not found"
    fi
done

python visualize_results.py model1.json model2.json model3.json --output ./charts/
```

### Debugging

Enable verbose output by modifying the script or adding debug prints:

```bash
# Add debugging to see what files are being processed
python -c "
import sys
print('Files to process:')
for f in sys.argv[1:]:
    if f.endswith('.json'):
        print(f'  {f}')
" "$@"

python visualize_results.py "$@"
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'matplotlib'
pip install matplotlib pandas seaborn numpy
```

#### 2. Permission Errors
```bash
# Error: Permission denied writing to output directory
chmod 755 ./charts/
# or
sudo python visualize_results.py results.json --output /var/www/charts/
```

#### 3. Memory Issues with Large Files
```bash
# For very large JSON files, process in smaller batches
python visualize_results.py batch1_*.json --output ./batch1/
python visualize_results.py batch2_*.json --output ./batch2/
```

#### 4. No Charts Generated
```bash
# Check if JSON files contain valid benchmark data
python -c "
import json
with open('your_file.json', 'r') as f:
    data = json.load(f)
    print('Keys:', list(data.keys()))
    if 'benchmarks' in data:
        print('Benchmarks:', list(data['benchmarks'].keys()))
"
```

#### 5. Timestamp Issues
```bash
# If timestamps are malformed, the script uses file modification time
# You can manually fix timestamps in JSON files:
sed -i 's/"timestamp": "malformed"/"timestamp": "2024-01-15T10:30:00"/' results.json
```

### Performance Tips

1. **Large Numbers of Files**: Process in batches
2. **High-Resolution Charts**: Reduce DPI if file sizes are too large
3. **Memory Usage**: Close chart windows if running interactively
4. **Faster Processing**: Use SSD storage for JSON files

### Getting Help

1. **Basic Help**: `python visualize_results.py --help`
2. **Debug Mode**: Add print statements to see data processing
3. **Check Input Data**: Validate JSON structure matches expected format
4. **File Permissions**: Ensure read access to input files and write access to output directory

## Version Information

- **Script Version**: 2.0
- **Python Compatibility**: 3.7+
- **Dependencies**: matplotlib, pandas, seaborn, numpy
- **Output Format**: PNG (300 DPI)
- **Chart Types**: 8 different visualization types
- **Supported Benchmarks**: 6 major LLM benchmarks