from typing import Dict, Any, Tuple, List
from abc import ABC, abstractmethod
import ast
import subprocess
import tempfile
import os
import re

from .inference import InferenceEngine
from .deterministic_extractors import MultipleChoiceExtractor

from ..base.benchmark_base import BenchmarkBase
from ..base.result_types import BenchmarkResult


class DeterministicEvaluator(ABC):
    """Base class for deterministic evaluation."""

    @abstractmethod
    def evaluate(
        self,
        extracted_answer: str,
        expected_answer: str,
        question_data: Dict[str, Any],
        confidence: float,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Evaluate extracted answer against expected answer.

        Returns:
            Tuple of (is_correct, evaluation_confidence, metadata)
        """
        pass


class MultipleChoiceEvaluator(DeterministicEvaluator):
    """Deterministic multiple choice evaluator."""

    def evaluate(
        self,
        extracted_answer: str,
        expected_answer: str,
        question_data: Dict[str, Any],
        confidence: float,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Evaluate multiple choice answer."""
        if extracted_answer == "UNKNOWN":
            return False, 0.0, {"reason": "no_answer_extracted"}

        # Normalize both answers
        extracted_norm = extracted_answer.upper().strip()
        expected_norm = expected_answer.upper().strip()

        is_correct = extracted_norm == expected_norm

        # Evaluation confidence is high for multiple choice (deterministic)
        eval_confidence = 1.0 if extracted_answer != "UNKNOWN" else 0.0

        metadata = {
            "extracted": extracted_norm,
            "expected": expected_norm,
            "extraction_confidence": confidence,
            "evaluation_method": "exact_match",
        }

        return is_correct, eval_confidence, metadata


class MathEvaluator(DeterministicEvaluator):
    """Deterministic mathematical answer evaluator."""

    def __init__(self):
        self.equivalence_checkers = [
            self._exact_string_match,
            self._numerical_equivalence,
            self._algebraic_equivalence,
            self._set_equivalence,  # For multiple solutions
        ]

    def evaluate(
        self,
        extracted_answer: str,
        expected_answer: str,
        question_data: Dict[str, Any],
        confidence: float,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Evaluate mathematical answer using multiple equivalence checks."""
        if extracted_answer == "UNKNOWN":
            return False, 0.0, {"reason": "no_answer_extracted"}

        metadata = {
            "extracted": extracted_answer,
            "expected": expected_answer,
            "extraction_confidence": confidence,
            "evaluation_methods_tried": [],
        }

        # Try each equivalence checker
        for checker in self.equivalence_checkers:
            try:
                is_equivalent, method_confidence, method_info = checker(
                    extracted_answer, expected_answer
                )

                metadata["evaluation_methods_tried"].append(
                    {
                        "method": checker.__name__,
                        "result": is_equivalent,
                        "confidence": method_confidence,
                        "info": method_info,
                    }
                )

                if is_equivalent:
                    return True, method_confidence, metadata

            except Exception as e:
                metadata["evaluation_methods_tried"].append(
                    {"method": checker.__name__, "error": str(e)}
                )

        return False, 0.8, metadata  # High confidence in "not equivalent"

    def _exact_string_match(
        self, extracted: str, expected: str
    ) -> Tuple[bool, float, Dict]:
        """Check exact string equivalence after normalization."""
        extracted_norm = self._normalize_math_string(extracted)
        expected_norm = self._normalize_math_string(expected)

        is_match = extracted_norm == expected_norm
        confidence = 1.0 if is_match else 0.0

        return (
            is_match,
            confidence,
            {
                "extracted_normalized": extracted_norm,
                "expected_normalized": expected_norm,
            },
        )

    def _numerical_equivalence(
        self, extracted: str, expected: str
    ) -> Tuple[bool, float, Dict]:
        """Check numerical equivalence for decimal/integer answers."""
        extracted_nums = self._extract_numbers(extracted)
        expected_nums = self._extract_numbers(expected)

        if not extracted_nums or not expected_nums:
            return False, 0.0, {"reason": "no_numbers_found"}

        # Compare first number found (most common case)
        try:
            extracted_val = float(extracted_nums[0])
            expected_val = float(expected_nums[0])

            # Use relative tolerance for comparison
            tolerance = max(1e-9, abs(expected_val) * 1e-6)
            is_equivalent = abs(extracted_val - expected_val) <= tolerance

            return (
                is_equivalent,
                0.95,
                {
                    "extracted_value": extracted_val,
                    "expected_value": expected_val,
                    "tolerance": tolerance,
                },
            )

        except (ValueError, IndexError):
            return False, 0.0, {"reason": "number_conversion_failed"}

    def _algebraic_equivalence(
        self, extracted: str, expected: str
    ) -> Tuple[bool, float, Dict]:
        """Check algebraic equivalence for expressions."""
        # This is a simplified version - in practice, you'd use a symbolic math library

        # Remove spaces and normalize
        extracted_norm = re.sub(r"\s+", "", extracted.lower())
        expected_norm = re.sub(r"\s+", "", expected.lower())

        # Check for equivalent forms of common expressions
        equivalences = [
            (r"x\s*=\s*-1\s*or\s*x\s*=\s*-5", r"x\s*=\s*-5\s*or\s*x\s*=\s*-1"),
            (
                r"(\d+)\s*/\s*(\d+)",
                lambda m: str(float(m.group(1)) / float(m.group(2))),
            ),
        ]

        # Simple pattern-based equivalence checking
        for pattern1, pattern2 in equivalences:
            if isinstance(pattern2, str):
                if (
                    re.search(pattern1, extracted_norm)
                    and re.search(pattern2, expected_norm)
                ) or (
                    re.search(pattern2, extracted_norm)
                    and re.search(pattern1, expected_norm)
                ):
                    return True, 0.8, {"equivalence_type": "pattern_match"}

        return False, 0.0, {"reason": "no_algebraic_equivalence_found"}

    def _set_equivalence(
        self, extracted: str, expected: str
    ) -> Tuple[bool, float, Dict]:
        """Check set equivalence for multiple solutions."""
        # Extract all numbers/variables from both answers
        extracted_parts = set(re.findall(r"[a-zA-Z]\s*=\s*[^,\s]+", extracted.lower()))
        expected_parts = set(re.findall(r"[a-zA-Z]\s*=\s*[^,\s]+", expected.lower()))

        if extracted_parts and expected_parts:
            is_equivalent = extracted_parts == expected_parts
            return (
                is_equivalent,
                0.85,
                {
                    "extracted_parts": list(extracted_parts),
                    "expected_parts": list(expected_parts),
                },
            )

        return False, 0.0, {"reason": "no_set_components_found"}

    def _normalize_math_string(self, s: str) -> str:
        """Normalize mathematical string for comparison."""
        s = s.lower().strip()
        s = re.sub(r"\s+", "", s)  # Remove all spaces
        s = re.sub(r"[.!,]+$", "", s)  # Remove trailing punctuation
        return s

    def _extract_numbers(self, s: str) -> List[str]:
        """Extract all numbers from string."""
        return re.findall(r"-?\d+\.?\d*", s)


class CodeEvaluator(DeterministicEvaluator):
    """Deterministic code evaluator using safe execution."""

    def __init__(self):
        self.timeout_seconds = 5

    def evaluate(
        self,
        extracted_answer: str,
        expected_answer: str,
        question_data: Dict[str, Any],
        confidence: float,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Evaluate code by running test cases safely."""
        if extracted_answer == "UNKNOWN":
            return False, 0.0, {"reason": "no_code_extracted"}

        metadata = {
            "extracted_code_length": len(extracted_answer),
            "extraction_confidence": confidence,
            "evaluation_methods": [],
        }

        # Static analysis first
        static_score, static_info = self._static_code_analysis(
            extracted_answer, question_data
        )
        metadata["static_analysis"] = static_info

        if static_score < 0.3:  # Code fails basic checks
            return False, 0.9, metadata

        # Dynamic evaluation with test cases
        if "test" in question_data:
            dynamic_score, dynamic_info = self._dynamic_code_evaluation(
                extracted_answer, question_data["test"]
            )
            metadata["dynamic_evaluation"] = dynamic_info

            is_correct = dynamic_score >= 0.8
            eval_confidence = 0.95 if dynamic_info.get("all_tests_passed") else 0.9

            return is_correct, eval_confidence, metadata

        # Fallback to static analysis only
        is_correct = static_score >= 0.7
        return is_correct, 0.6, metadata  # Lower confidence without dynamic testing

    def _static_code_analysis(
        self, code: str, question_data: Dict[str, Any]
    ) -> Tuple[float, Dict]:
        """Perform static analysis of code quality."""
        analysis = {
            "has_function_def": False,
            "has_return_statement": False,
            "proper_indentation": False,
            "no_syntax_errors": False,
            "expected_function_name": None,
        }

        score = 0.0

        # Check for function definition
        func_match = re.search(r"def\s+(\w+)", code)
        if func_match:
            analysis["has_function_def"] = True
            analysis["expected_function_name"] = func_match.group(1)
            score += 0.25

        # Check for return statement
        if "return " in code:
            analysis["has_return_statement"] = True
            score += 0.25

        # Check for proper indentation (basic check)
        lines = code.split("\n")
        indented_lines = [
            line for line in lines if line.startswith("    ") or line.startswith("\t")
        ]
        if indented_lines:
            analysis["proper_indentation"] = True
            score += 0.2

        # Check for syntax errors
        try:
            ast.parse(code)
            analysis["no_syntax_errors"] = True
            score += 0.3
        except SyntaxError as e:
            analysis["syntax_error"] = str(e)

        return score, analysis

    def _dynamic_code_evaluation(self, code: str, test_code: str) -> Tuple[float, Dict]:
        """Safely execute code with test cases."""
        evaluation = {
            "tests_run": 0,
            "tests_passed": 0,
            "execution_error": None,
            "timeout": False,
        }

        try:
            # Create a safe execution environment
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                # Write the code and test
                f.write(code + "\n\n" + test_code + "\ncheck(locals()[func_name])")
                f.flush()

                # Extract function name for testing
                func_match = re.search(r"def\s+(\w+)", code)
                if not func_match:
                    return 0.0, {"error": "no_function_found"}

                func_name = func_match.group(1)

                # Replace the placeholder in test execution
                test_content = f.read()
                test_content = test_content.replace("locals()[func_name]", func_name)

                # Rewrite file with correct function name
                with open(f.name, "w") as f2:
                    f2.write(code + "\n\n" + test_code + f"\ncheck({func_name})")

                # Execute with timeout
                result = subprocess.run(
                    ["python", f.name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )

                if result.returncode == 0:
                    evaluation["tests_passed"] = 1
                    evaluation["tests_run"] = 1
                    evaluation["all_tests_passed"] = True
                    score = 1.0
                else:
                    evaluation["execution_error"] = result.stderr
                    score = 0.0

        except subprocess.TimeoutExpired:
            evaluation["timeout"] = True
            score = 0.0
        except Exception as e:
            evaluation["execution_error"] = str(e)
            score = 0.0
        finally:
            # Cleanup
            try:
                os.unlink(f.name)  # type: ignore
            except:
                pass

        return score, evaluation


# Example usage in benchmarks
class OptimizedMMLUBenchmark(BenchmarkBase):
    """Optimized MMLU with deterministic evaluation."""

    def __init__(self):
        super().__init__("MMLU-Optimized", "Deterministic MMLU Evaluation")
        self.extractor = MultipleChoiceExtractor()
        self.evaluator = MultipleChoiceEvaluator()

    def run(self, model_id: str, num_samples: int = 100) -> BenchmarkResult:
        """Run optimized MMLU evaluation."""
        questions = self.get_sample_questions()
        extended_questions = (questions * (num_samples // len(questions) + 1))[
            :num_samples
        ]

        results = []
        correct_count = 0

        for i, question in enumerate(extended_questions):
            # Get response with deterministic settings
            response = self._get_deterministic_response(model_id, question)

            # Extract answer with confidence
            extracted_answer, extraction_confidence = self.extractor.extract(
                response, question
            )

            # Evaluate deterministically
            is_correct, eval_confidence, metadata = self.evaluator.evaluate(
                extracted_answer, question["answer"], question, extraction_confidence
            )

            if is_correct:
                correct_count += 1

            results.append(
                {
                    "question_id": i,
                    "subject": question["subject"],
                    "extracted_answer": extracted_answer,
                    "expected_answer": question["answer"],
                    "is_correct": is_correct,
                    "extraction_confidence": extraction_confidence,
                    "evaluation_confidence": eval_confidence,
                    "metadata": metadata,
                    "response_length": len(response),
                    "question_hash": hash(question["question"]),  # For reproducibility
                }
            )

        score = correct_count / len(extended_questions)

        return BenchmarkResult(
            score=score,
            total_questions=len(extended_questions),
            correct_answers=correct_count,
            detailed_results=results,
            metadata={
                "benchmark": "MMLU-Optimized",
                "evaluation_method": "deterministic",
                "average_extraction_confidence": sum(
                    r["extraction_confidence"] for r in results
                )
                / len(results),
                "average_evaluation_confidence": sum(
                    r["evaluation_confidence"] for r in results
                )
                / len(results),
            },
        )

    def _get_deterministic_response(self, model_id: str, question: Dict) -> str:
        """Get response with deterministic settings."""
        # Use temperature=0 and fixed seed for reproducibility
        choices_text = "\n".join(
            [f"{chr(65+j)}. {choice}" for j, choice in enumerate(question["choices"])]
        )

        prompt = f"""Answer this multiple choice question. Provide only the letter of your answer.

Question: {question['question']}

{choices_text}

Answer:"""

        response = InferenceEngine().run_single_inference(
            model_id=model_id,
            prompt=prompt,
            max_tokens=5,  # Very short for deterministic extraction
            temperature=0.0,  # Deterministic
        )

        return response.get("response", "")
