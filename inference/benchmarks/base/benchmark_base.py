import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from .result_types import BenchmarkResult


class BenchmarkBase(ABC):
    """Abstract base class for all benchmarks."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"benchmark.{name}")

    @abstractmethod
    def run(self, model_id: str, num_samples: int = 50) -> BenchmarkResult:
        """Run the benchmark on the specified model."""
        pass

    @abstractmethod
    def get_sample_questions(self) -> List[Dict[str, Any]]:
        """Get sample questions for the benchmark."""
        pass

    def validate_sample_count(self, num_samples: int, max_samples: int = -1) -> int:
        """Validate and adjust sample count."""
        if max_samples and num_samples > max_samples:
            self.logger.warning(
                f"Requested {num_samples} samples, but max is {max_samples}"
            )
            return max_samples
        return max(1, num_samples)
