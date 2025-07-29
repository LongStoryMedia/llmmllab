from typing import List, Dict, Any
import logging
from ..base.benchmark_base import BenchmarkBase
from ..base.result_types import BenchmarkResult
from ..utils.inference import InferenceEngine
from ..utils.deterministic_evaluators import MathEvaluator


class Math500Benchmark(BenchmarkBase):
    """MATH-500 benchmark - Mathematical problem solving."""

    def __init__(self):
        super().__init__(
            name="MATH-500",
            description="Mathematical problem solving across multiple domains",
        )
        self.inference_engine = InferenceEngine()
        self.evaluator = MathEvaluator()
        self.logger = logging.getLogger("benchmark.math500")

    def get_sample_questions(self) -> List[Dict[str, Any]]:
        """Get sample MATH problems."""
        return [
            {
                "problem": "If $x^2 + 6x + 5 = 0$, what are the values of $x$?",
                "solution": "Using the quadratic formula or factoring: $(x+1)(x+5) = 0$, so $x = -1$ or $x = -5$.",
                "answer": "x = -1 or x = -5",
                "level": "Level 2",
                "type": "Algebra",
            },
            {
                "problem": "What is the area of a triangle with vertices at $(0,0)$, $(5,0)$, and $(0,3)$?",
                "solution": "This is a right triangle with base 5 and height 3. Area = (1/2) * base * height = (1/2) * 5 * 3 = 7.5",
                "answer": "7.5",
                "level": "Level 1",
                "type": "Geometry",
            },
            {
                "problem": "Find the derivative of $f(x) = 3x^4 - 2x^2 + 7x - 1$.",
                "solution": "Using power rule: $f'(x) = 12x^3 - 4x + 7$",
                "answer": "12x^3 - 4x + 7",
                "level": "Level 2",
                "type": "Calculus",
            },
        ]

    def run(self, model_id: str, num_samples: int = 50) -> BenchmarkResult:
        """Run MATH-500 benchmark."""
        print(f"\n--- MATH-500 Benchmark ---")

        problems = self.get_sample_questions()
        num_samples = self.validate_sample_count(num_samples)
        extended_problems = (problems * (num_samples // len(problems) + 1))[
            :num_samples
        ]

        correct_answers = 0
        detailed_results = []

        for i, problem in enumerate(extended_problems):
            print(f"MATH Problem {i+1}/{len(extended_problems)}")

            prompt = f"""Solve the following mathematical problem step by step. Provide your final answer clearly.

Problem: {problem['problem']}

Solution:"""

            try:
                print(f"\n{'-'*80}")
                print(f"QUESTION: {problem['problem']}")
                print(f"EXPECTED ANSWER: {problem['answer']}")

                response = self.inference_engine.run_single_inference(
                    model_id, prompt, max_tokens=300, temperature=0.1
                )
                model_response = response.get("response", "")

                self.logger.info(f"\nMODEL RESPONSE:")
                self.logger.info(
                    f"{model_response[:800]}{'...' if len(model_response) > 800 else ''}"
                )

                # Use the MathEvaluator's evaluate method instead
                is_correct, confidence, eval_metadata = self.evaluator.evaluate(
                    model_response, problem["answer"], problem, 1.0
                )

                self.logger.info(f"\nEVALUATION:")
                self.logger.info(f"Expected answer: {problem['answer']}")
                self.logger.info(
                    f"Is correct: {'YES' if is_correct else 'NO'} (confidence: {confidence:.2f})"
                )

                if is_correct:
                    correct_answers += 1

                detailed_results.append(
                    {
                        "problem_id": i,
                        "type": problem["type"],
                        "level": problem["level"],
                        "expected_answer": problem["answer"],
                        "is_correct": is_correct,
                        "confidence": confidence,
                        "evaluation_metadata": eval_metadata,
                        "response": (
                            model_response[:200] + "..."
                            if len(model_response) > 200
                            else model_response
                        ),
                    }
                )

            except Exception as e:
                self.logger.error(f"Error in MATH problem {i}: {str(e)}")

        math_score = (
            correct_answers / len(extended_problems) if extended_problems else 0
        )
        self.logger.info(
            f"MATH-500 Score: {math_score:.3f} ({correct_answers}/{len(extended_problems)})"
        )

        return BenchmarkResult(
            score=math_score,
            total_questions=len(extended_problems),
            correct_answers=correct_answers,
            detailed_results=detailed_results[:5],
            metadata={
                "benchmark": "MATH-500",
                "description": "Mathematical problem solving across multiple domains",
            },
        )
