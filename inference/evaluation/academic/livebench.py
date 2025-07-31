import datetime
from typing import List, Dict, Any
from ..base.benchmark_base import BenchmarkBase
from ..base.result_types import BenchmarkResult
from ..utils.inference import InferenceEngine
from ..utils.prompt_templates import PromptTemplates


class LiveBenchmark(BenchmarkBase):
    """
    LiveBench implementation - Contamination-free evaluation with regularly updated questions.

    Note: LiveBench questions are updated monthly to prevent data contamination.
    This is a simplified implementation - the real LiveBench would pull fresh questions.
    """

    def __init__(self):
        super().__init__(
            name="LiveBench",
            description="Contamination-free evaluation with fresh questions",
        )
        self.inference_engine = InferenceEngine()
        # LiveBench uses custom evaluators for each question type

    def get_sample_questions(self) -> List[Dict[str, Any]]:
        """Get sample LiveBench questions."""
        return [
            {
                "category": "reasoning",
                "question": "A new social media platform launched in January 2024. If it gained 1 million users in the first month, then doubled its user base each subsequent month, how many users would it have by June 2024?",
                "expected_reasoning": ["exponential growth", "doubling", "6 months"],
                "type": "math_reasoning",
            },
            {
                "category": "knowledge",
                "question": "What are the key provisions of the EU AI Act that was finalized in 2024?",
                "expected_concepts": [
                    "artificial intelligence",
                    "regulation",
                    "risk-based approach",
                    "prohibited AI",
                ],
                "type": "current_events",
            },
            {
                "category": "instruction_following",
                "question": "Create a haiku about quantum computing. The first line must contain exactly 5 syllables, the second line 7 syllables, and the third line 5 syllables.",
                "constraints": [
                    "haiku format",
                    "quantum computing theme",
                    "syllable count",
                ],
                "type": "creative_constraint",
            },
        ]

    def run(self, model_id: str, num_samples: int = 10) -> BenchmarkResult:
        """Run LiveBench evaluation."""
        print("\n--- LiveBench Evaluation ---")

        questions = self.get_sample_questions()
        # LiveBench typically uses fewer samples due to fresh content
        num_samples = min(num_samples, len(questions) * 3)
        extended_questions = (questions * (num_samples // len(questions) + 1))[
            :num_samples
        ]

        total_score = 0
        detailed_results = []

        for i, question in enumerate(extended_questions):
            print(
                f"LiveBench Question {i+1}/{len(extended_questions)} ({question['category']})"
            )

            try:
                # Use livebench_template from PromptTemplates
                prompt = PromptTemplates.livebench_template(
                    question=question["question"], category=question["category"]
                )
                response = self.inference_engine.run_single_inference(
                    model_id, prompt, max_tokens=200, temperature=0.3
                )
                model_response = response.get("response", "")

                # Evaluate based on question type
                if question["type"] == "math_reasoning":
                    score = self._evaluate_math_reasoning(model_response, question)
                elif question["type"] == "current_events":
                    score = self._evaluate_knowledge(model_response, question)
                elif question["type"] == "creative_constraint":
                    score = self._evaluate_creative_constraint(model_response, question)
                else:
                    score = 0.5  # Default middle score

                total_score += score

                detailed_results.append(
                    {
                        "question_id": i,
                        "category": question["category"],
                        "type": question["type"],
                        "score": score,
                        "response": (
                            model_response[:150] + "..."
                            if len(model_response) > 150
                            else model_response
                        ),
                    }
                )

                print(f"  Score: {score:.2f}")

            except Exception as e:
                self.logger.error(f"Error in LiveBench question {i}: {str(e)}")

        livebench_score = (
            total_score / len(extended_questions) if extended_questions else 0
        )
        print(f"LiveBench Score: {livebench_score:.3f}")

        return BenchmarkResult(
            score=livebench_score,
            total_questions=len(extended_questions),
            correct_answers=int(total_score),  # Approximate since scores are continuous
            detailed_results=detailed_results,
            metadata={
                "benchmark": "LiveBench",
                "description": "Contamination-free evaluation with fresh questions",
                "evaluation_date": datetime.datetime.now().isoformat(),
            },
        )

    def _evaluate_math_reasoning(self, response: str, question: Dict) -> float:
        """Evaluate mathematical reasoning response."""
        response_lower = response.lower()

        # Check for correct final answer (32 million for the doubling question)
        if "32" in response and ("million" in response_lower or "000000" in response):
            return 1.0
        elif any(
            term in response_lower for term in ["double", "exponential", "growth"]
        ):
            return 0.6  # Partial credit for showing understanding
        else:
            return 0.0

    def _evaluate_knowledge(self, response: str, question: Dict) -> float:
        """Evaluate knowledge-based response."""
        response_lower = response.lower()
        expected_concepts = question.get("expected_concepts", [])

        concepts_found = sum(
            1 for concept in expected_concepts if concept.lower() in response_lower
        )
        return concepts_found / len(expected_concepts) if expected_concepts else 0.0

    def _evaluate_creative_constraint(self, response: str, question: Dict) -> float:
        """Evaluate creative writing with constraints."""
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]

        if len(lines) != 3:
            return 0.0

        # Simple syllable counting (very approximate)
        def count_syllables(text):
            vowels = "aeiouy"
            text = text.lower()
            count = 0
            prev_char_was_vowel = False

            for char in text:
                if char in vowels:
                    if not prev_char_was_vowel:
                        count += 1
                    prev_char_was_vowel = True
                else:
                    prev_char_was_vowel = False

            return max(1, count)  # At least 1 syllable per word

        syllables = [count_syllables(line) for line in lines]

        # Check if it follows 5-7-5 pattern (with some tolerance)
        if 4 <= syllables[0] <= 6 and 6 <= syllables[1] <= 8 and 4 <= syllables[2] <= 6:
            # Check if it mentions quantum computing
            if any(
                term in response.lower()
                for term in ["quantum", "qubit", "superposition", "entangle"]
            ):
                return 1.0
            else:
                return 0.7  # Good format but missing theme
        else:
            return 0.3  # Poor format
