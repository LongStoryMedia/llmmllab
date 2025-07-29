# LLM Benchmarks

A framework for benchmarking large language models (LLMs) on various academic and practical tasks.

## Overview

This benchmark framework allows for consistent evaluation of language models across different tasks:

- **Academic benchmarks**: Standard NLP benchmarks like HumanEval, MMLU, etc.
- **Practical benchmarks**: Real-world application tasks
- **Customizable evaluation**: Extensible framework for adding new benchmarks

## Architecture

The benchmarking system is organized as follows:

```text
inference/benchmarks/
├── academic/          # Academic benchmarks
│   └── humaneval.py   # HumanEval code generation benchmark
├── base/              # Base classes and common types
│   ├── benchmark_base.py  # Abstract base class for benchmarks
│   └── result_types.py    # Standard result formats
├── config/            # Configuration for benchmarks
├── utils/             # Utility functions and classes
│   ├── evaluators.py       # Answer evaluation tools
│   ├── inference.py        # Model inference helpers
│   └── deterministic_evaluators.py  # Deterministic evaluation methods
└── main.py            # CLI entry point
```

## Getting Started

### Prerequisites

- Python 3.8+
- Access to model inference APIs or local models
- Dependencies listed in `requirements.txt`

### Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r inference/requirements.txt
```

### Running Benchmarks

To run the benchmark suite:

```bash
python -m inference.benchmarks.main --models model_id1 model_id2 --benchmarks humaneval mmlu
```

#### Command Line Options

- `--models`: One or more model IDs to evaluate
- `--benchmarks`: Specific benchmarks to run (default: all)
- `--samples`: Number of samples per benchmark (default: 50)
- `--output`: Output file path for results (JSON format)
- `--csv`: Generate CSV summary in addition to JSON
- `--data-dir`: Directory for benchmark data

## Benchmarks

### HumanEval

Tests a model's ability to generate functionally correct code based on docstrings and function signatures.

Example:

```python
from inference.benchmarks.academic.humaneval import HumanEvalBenchmark

benchmark = HumanEvalBenchmark()
result = benchmark.run("model_id", num_samples=20)
print(f"HumanEval Score: {result.score}")
```

### Adding New Benchmarks

To add a new benchmark:

1. Create a new class that inherits from `BenchmarkBase`
2. Implement the required methods:
   - `get_sample_questions()`
   - `run(model_id, num_samples)`
3. Register the benchmark in the benchmark suite

Example skeleton:

```python
from typing import List, Dict, Any
from ..base.benchmark_base import BenchmarkBase
from ..base.result_types import BenchmarkResult

class NewBenchmark(BenchmarkBase):
    def __init__(self):
        super().__init__(
            name="NewBenchmark", 
            description="Description of the benchmark"
        )
    
    def get_sample_questions(self) -> List[Dict[str, Any]]:
        # Return a list of sample questions
        return [...]
    
    def run(self, model_id: str, num_samples: int = 50) -> BenchmarkResult:
        # Implement benchmark logic
        # Return BenchmarkResult with scores
```

## Development

### Testing

Run tests:

```bash
python -m pytest inference/benchmarks/tests/
```

### Extending Evaluators

For custom evaluation logic, extend the appropriate evaluator classes in `utils/evaluators.py` or `utils/deterministic_evaluators.py`.

## License

This project is licensed under the terms included in the LICENSE file.
