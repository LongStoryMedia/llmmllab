from typing import List, Dict, Any, Optional
import logging
from ..base.benchmark_base import BenchmarkBase
from ..base.result_types import BenchmarkResult
from ..utils.inference import InferenceEngine
from ..utils.deterministic_evaluators import CodeEvaluator
from ..utils.prompt_templates import PromptTemplates


class HumanEvalBenchmark(BenchmarkBase):
    """HumanEval benchmark - Code generation evaluation."""

    def __init__(self):
        super().__init__(
            name="HumanEval", description="Code generation and functional correctness"
        )
        self.inference_engine = InferenceEngine()
        self.evaluator = CodeEvaluator()
        self.logger = logging.getLogger("benchmark.humaneval")

    def get_sample_questions(self) -> List[Dict[str, Any]]:
        """Get sample HumanEval problems."""
        return [
            {
                "task_id": "HumanEval/0",
                "prompt": '''def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """''',
                "canonical_solution": """    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False""",
                "test": """def check(candidate):
    assert candidate([1.0, 2.0, 3.0], 0.5) == False
    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True""",
            },
            {
                "task_id": "HumanEval/1",
                "prompt": '''def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """''',
                "canonical_solution": """    result = []
    current_string = []
    current_depth = 0

    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string.append(c)
        elif c == ')':
            current_depth -= 1
            current_string.append(c)

            if current_depth == 0:
                result.append(''.join(current_string))
                current_string = []

    return result""",
                "test": """def check(candidate):
    assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']
    assert candidate('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']""",
            },
        ]

    def run(
        self, model_id: str, num_samples: int = 50, dataset_path: Optional[str] = None
    ) -> BenchmarkResult:
        """Run HumanEval benchmark."""
        self.logger.info(f"\n--- HumanEval Benchmark ---")

        problems = self.get_sample_questions()
        num_samples = self.validate_sample_count(num_samples)
        extended_problems = (problems * (num_samples // len(problems) + 1))[
            :num_samples
        ]

        correct_solutions = 0
        detailed_results = []

        for i, problem in enumerate(extended_problems):
            self.logger.info(f"HumanEval Problem {i+1}/{len(extended_problems)}")

            # Use the code_template from PromptTemplates
            prompt = PromptTemplates.code_template(problem["prompt"])

            try:
                response = self.inference_engine.run_single_inference(
                    model_id, prompt, max_tokens=400, temperature=0.1
                )
                model_code = response.get("response", "")

                # Use the deterministic CodeEvaluator to check correctness
                is_correct, confidence, eval_metadata = self.evaluator.evaluate(
                    model_code, "", problem, 1.0
                )

                if is_correct:
                    correct_solutions += 1

                detailed_results.append(
                    {
                        "task_id": problem["task_id"],
                        "is_correct": is_correct,
                        "confidence": confidence,
                        "generated_code": (
                            model_code[:200] + "..."
                            if len(model_code) > 200
                            else model_code
                        ),
                        "evaluation_metadata": eval_metadata,
                    }
                )

            except Exception as e:
                self.logger.error(f"Error in HumanEval problem {i}: {str(e)}")

        humaneval_score = (
            correct_solutions / len(extended_problems) if extended_problems else 0
        )
        self.logger.info(
            f"HumanEval Score: {humaneval_score:.3f} ({correct_solutions}/{len(extended_problems)})"
        )

        return BenchmarkResult(
            score=humaneval_score,
            total_questions=len(extended_problems),
            correct_answers=correct_solutions,
            detailed_results=detailed_results[
                :5
            ],  # Limit detailed results to avoid excessive output
            metadata={
                "benchmark": "HumanEval",
                "description": "Code generation and functional correctness",
                "model_id": model_id,
                "temperature": 0.1,
                "max_tokens": 400,
            },
        )
