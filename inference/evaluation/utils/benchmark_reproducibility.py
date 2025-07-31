import hashlib
import json
from typing import Dict, Any, List
import random


class ReproducibilityManager:
    """Ensure benchmark results are reproducible."""

    @staticmethod
    def create_question_hash(question: Dict[str, Any]) -> str:
        """Create a deterministic hash for a question."""
        # Create a canonical representation
        canonical = {
            "question": question.get("question", ""),
            "choices": question.get("choices", []),
            "answer": question.get("answer", ""),
            "subject": question.get("subject", ""),
        }

        canonical_str = json.dumps(canonical, sort_keys=True)
        return hashlib.md5(canonical_str.encode()).hexdigest()

    @staticmethod
    def create_evaluation_hash(
        extracted: str, expected: str, method: str, confidence: float
    ) -> str:
        """Create a hash for evaluation reproducibility."""
        eval_data = {
            "extracted": extracted,
            "expected": expected,
            "method": method,
            "confidence": round(confidence, 6),  # Round to avoid floating point issues
        }

        eval_str = json.dumps(eval_data, sort_keys=True)
        return hashlib.md5(eval_str.encode()).hexdigest()

    @staticmethod
    def validate_reproducibility(results1: List[Dict], results2: List[Dict]) -> Dict:
        """Compare two benchmark runs for reproducibility."""
        if len(results1) != len(results2):
            return {"reproducible": False, "reason": "Different number of results"}

        mismatches = []
        for i, (r1, r2) in enumerate(zip(results1, results2)):
            if r1.get("question_hash") != r2.get("question_hash"):
                mismatches.append(f"Question {i}: Different question hashes")
            if r1.get("is_correct") != r2.get("is_correct"):
                mismatches.append(f"Question {i}: Different correctness results")
            if (
                abs(
                    r1.get("extraction_confidence", 0)
                    - r2.get("extraction_confidence", 0)
                )
                > 1e-6
            ):
                mismatches.append(f"Question {i}: Different extraction confidence")

        return {
            "reproducible": len(mismatches) == 0,
            "mismatches": mismatches,
            "match_rate": 1.0 - len(mismatches) / len(results1),
        }
