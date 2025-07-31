import logging
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import numpy as np


class QualityAssuranceMonitor:
    """Monitor and report on benchmark quality metrics."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.extraction_stats = defaultdict(list)
        self.evaluation_stats = defaultdict(list)
        self.error_log = []

    def record_extraction(
        self,
        benchmark_name: str,
        question_id: str,
        extracted: str,
        confidence: float,
        metadata: Dict,
    ):
        """Record extraction statistics."""
        self.extraction_stats[benchmark_name].append(
            {
                "question_id": question_id,
                "extracted": extracted,
                "confidence": confidence,
                "success": extracted != "UNKNOWN",
                "metadata": metadata,
            }
        )

    def record_evaluation(
        self,
        benchmark_name: str,
        question_id: str,
        is_correct: bool,
        eval_confidence: float,
        metadata: Dict,
    ):
        """Record evaluation statistics."""
        self.evaluation_stats[benchmark_name].append(
            {
                "question_id": question_id,
                "is_correct": is_correct,
                "eval_confidence": eval_confidence,
                "metadata": metadata,
            }
        )

    def record_error(
        self, benchmark_name: str, question_id: str, error_type: str, error_message: str
    ):
        """Record errors for analysis."""
        self.error_log.append(
            {
                "benchmark_name": benchmark_name,
                "question_id": question_id,
                "error_type": error_type,
                "error_message": error_message,
            }
        )

    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        report = {
            "extraction_quality": self._analyze_extraction_quality(),
            "evaluation_quality": self._analyze_evaluation_quality(),
            "error_analysis": self._analyze_errors(),
            "recommendations": self._generate_recommendations(),
        }

        return report

    def _analyze_extraction_quality(self) -> Dict[str, Any]:
        """Analyze extraction quality metrics."""
        analysis = {}

        for benchmark_name, extractions in self.extraction_stats.items():
            if not extractions:
                continue

            success_rate = sum(1 for e in extractions if e["success"]) / len(
                extractions
            )
            confidences = [e["confidence"] for e in extractions if e["success"]]

            analysis[benchmark_name] = {
                "total_extractions": len(extractions),
                "success_rate": success_rate,
                "avg_confidence": np.mean(confidences) if confidences else 0.0,
                "confidence_std": np.std(confidences) if confidences else 0.0,
                "low_confidence_count": sum(1 for c in confidences if c < 0.5),
                "high_confidence_count": sum(1 for c in confidences if c > 0.8),
            }

        return analysis

    def _analyze_evaluation_quality(self) -> Dict[str, Any]:
        """Analyze evaluation quality metrics."""
        analysis = {}

        for benchmark_name, evaluations in self.evaluation_stats.items():
            if not evaluations:
                continue

            accuracy = sum(1 for e in evaluations if e["is_correct"]) / len(evaluations)
            eval_confidences = [e["eval_confidence"] for e in evaluations]

            analysis[benchmark_name] = {
                "total_evaluations": len(evaluations),
                "accuracy": accuracy,
                "avg_eval_confidence": np.mean(eval_confidences),
                "eval_confidence_std": np.std(eval_confidences),
            }

        return analysis

    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze error patterns."""
        if not self.error_log:
            return {"total_errors": 0}

        error_types = Counter(error["error_type"] for error in self.error_log)
        error_by_benchmark = Counter(
            error["benchmark_name"] for error in self.error_log
        )

        return {
            "total_errors": len(self.error_log),
            "error_types": dict(error_types),
            "errors_by_benchmark": dict(error_by_benchmark),
            "error_rate": (
                len(self.error_log)
                / sum(
                    len(extractions) for extractions in self.extraction_stats.values()
                )
                if self.extraction_stats
                else 0.0
            ),
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on quality analysis."""
        recommendations = []

        # Analyze extraction quality
        for benchmark_name, extractions in self.extraction_stats.items():
            if not extractions:
                continue

            success_rate = sum(1 for e in extractions if e["success"]) / len(
                extractions
            )
            if success_rate < 0.8:
                recommendations.append(
                    f"{benchmark_name}: Low extraction success rate ({success_rate:.2%}). "
                    "Consider improving extraction patterns or prompt clarity."
                )

            confidences = [e["confidence"] for e in extractions if e["success"]]
            if confidences and np.mean(confidences) < 0.6:
                recommendations.append(
                    f"{benchmark_name}: Low average extraction confidence. "
                    "Review extraction logic and consider more reliable patterns."
                )

        # Analyze error patterns
        if self.error_log:
            error_types = Counter(error["error_type"] for error in self.error_log)
            most_common_error = error_types.most_common(1)[0]

            if most_common_error[1] > len(self.error_log) * 0.3:
                recommendations.append(
                    f"High frequency of {most_common_error[0]} errors. "
                    "Focus on addressing this error type."
                )

        if not recommendations:
            recommendations.append("Quality metrics look good! Continue monitoring.")

        return recommendations
