from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class BenchmarkResult:
    """Standard result format for academic benchmarks."""

    score: float
    total_questions: int
    correct_answers: int
    detailed_results: List[Dict]
    metadata: Dict[str, Any]
