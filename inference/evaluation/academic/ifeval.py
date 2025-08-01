from typing import List, Dict, Any, Callable, Optional
import logging
import re
from ..base.benchmark_base import BenchmarkBase
from ..base.result_types import BenchmarkResult
from ..utils.inference import InferenceEngine
from ..utils.prompt_templates import PromptTemplates


class IFEvalBenchmark(BenchmarkBase):
    """IFEVAL benchmark - Instruction Following Evaluation."""

    def __init__(self):
        super().__init__(name="IFEVAL", description="Instruction Following Evaluation")
        self.inference_engine = InferenceEngine()
        self.logger = logging.getLogger("benchmark.ifeval")

    def get_sample_questions(self) -> List[Dict[str, Any]]:
        """Get sample IFEVAL tasks."""
        return [
            {
                "instruction": "Write a poem about the ocean that is exactly 4 lines long.",
                "constraint": "exactly_4_lines",
                "constraint_params": {},
            },
            {
                "instruction": "List 5 countries in Europe. Format each as: Country: [name]",
                "constraint": "format_and_count",
                "constraint_params": {"count": 5},
            },
            {
                "instruction": "Explain photosynthesis in exactly two sentences.",
                "constraint": "exactly_2_sentences",
                "constraint_params": {},
            },
            {
                "instruction": "Write a response that contains the word 'fantastic' exactly 3 times.",
                "constraint": "word_frequency",
                "constraint_params": {"word": "fantastic", "count": 3},
            },
            {
                "instruction": "Create a numbered list with exactly 6 items about healthy eating habits.",
                "constraint": "numbered_list_6_items",
                "constraint_params": {},
            },
        ]

    def run(
        self, model_id: str, num_samples: int = 50, dataset_path: Optional[str] = None
    ) -> BenchmarkResult:
        """Run IFEVAL benchmark."""
        self.logger.info(f"\n--- IFEVAL Benchmark ---")

        tasks = self.get_sample_questions()
        num_samples = self.validate_sample_count(num_samples)
        extended_tasks = (tasks * (num_samples // len(tasks) + 1))[:num_samples]

        successful_instructions = 0
        detailed_results = []

        for i, task in enumerate(extended_tasks):
            self.logger.info(f"IFEVAL Task {i+1}/{len(extended_tasks)}")

            try:
                # Use instruction_following_template from PromptTemplates
                prompt = PromptTemplates.instruction_following_template(
                    task["instruction"]
                )
                response = self.inference_engine.run_single_inference(
                    model_id, prompt, max_tokens=200, temperature=0.3
                )
                model_response = response.get("response", "")

                # Check if instruction was followed using our constraint methods
                follows_instruction, confidence, metadata = (
                    self.check_instruction_following(
                        model_response, task["constraint"], **task["constraint_params"]
                    )
                )

                if follows_instruction:
                    successful_instructions += 1

                detailed_results.append(
                    {
                        "task_id": i,
                        "constraint": task["constraint"],
                        "follows_instruction": follows_instruction,
                        "confidence": confidence,
                        "metadata": metadata,
                        "response": (
                            model_response[:150] + "..."
                            if len(model_response) > 150
                            else model_response
                        ),
                    }
                )

            except Exception as e:
                self.logger.error(f"Error in IFEVAL task {i}: {str(e)}")

        ifeval_score = (
            successful_instructions / len(extended_tasks) if extended_tasks else 0
        )
        self.logger.info(
            f"IFEVAL Score: {ifeval_score:.3f} ({successful_instructions}/{len(extended_tasks)})"
        )

        return BenchmarkResult(
            score=ifeval_score,
            total_questions=len(extended_tasks),
            correct_answers=successful_instructions,
            detailed_results=detailed_results[:5],
            metadata={
                "benchmark": "IFEVAL",
                "description": "Instruction Following Evaluation",
            },
        )

    def check_instruction_following(
        self, response: str, constraint_type: str, **constraint_params
    ) -> tuple[bool, float, dict]:
        """
        Check if the model response follows specific constraints.

        Args:
            response: The model response text
            constraint_type: Type of constraint to check
            **constraint_params: Parameters specific to the constraint type

        Returns:
            Tuple of (follows_instruction, confidence, metadata)
        """
        if not response.strip():
            return False, 1.0, {"reason": "empty_response"}

        # Constraint checking methods
        if constraint_type == "exactly_4_lines":
            return self._check_line_count(response, 4)

        elif constraint_type == "exactly_2_sentences":
            return self._check_sentence_count(response, 2)

        elif constraint_type == "format_and_count":
            count = constraint_params.get("count", 5)
            return self._check_formatted_list(response, count)

        elif constraint_type == "word_frequency":
            word = constraint_params.get("word", "")
            count = constraint_params.get("count", 0)
            return self._check_word_frequency(response, word, count)

        elif constraint_type == "numbered_list_6_items":
            return self._check_numbered_list(response, 6)

        # Unknown constraint type
        return False, 0.0, {"reason": f"unknown_constraint_type: {constraint_type}"}

    def _check_line_count(
        self, response: str, expected_lines: int
    ) -> tuple[bool, float, dict]:
        """Check if response has exactly the specified number of non-empty lines."""
        lines = [line for line in response.split("\n") if line.strip()]
        is_correct = len(lines) == expected_lines
        confidence = 1.0  # High confidence in line count
        return (
            is_correct,
            confidence,
            {
                "expected_lines": expected_lines,
                "actual_lines": len(lines),
                "lines": lines,
            },
        )

    def _check_sentence_count(
        self, response: str, expected_sentences: int
    ) -> tuple[bool, float, dict]:
        """Check if response has exactly the specified number of sentences."""
        # Simple sentence splitting by common end punctuation
        sentences = re.split(r"[.!?]+", response)
        sentences = [s.strip() for s in sentences if s.strip()]
        is_correct = len(sentences) == expected_sentences
        confidence = (
            0.9  # High confidence but not perfect due to sentence boundary ambiguity
        )
        return (
            is_correct,
            confidence,
            {
                "expected_sentences": expected_sentences,
                "actual_sentences": len(sentences),
                "sentences": sentences,
            },
        )

    def _check_formatted_list(
        self, response: str, expected_count: int
    ) -> tuple[bool, float, dict]:
        """Check if response contains formatted list items."""
        pattern = r"(?:Country|country):\s+(\w+)"
        matches = re.findall(pattern, response)
        is_correct = len(matches) == expected_count
        confidence = 0.95
        return (
            is_correct,
            confidence,
            {
                "expected_count": expected_count,
                "actual_count": len(matches),
                "items": matches,
            },
        )

    def _check_word_frequency(
        self, response: str, word: str, expected_count: int
    ) -> tuple[bool, float, dict]:
        """Check if a specific word appears exactly the expected number of times."""
        lower_response = response.lower()
        lower_word = word.lower()
        # Match whole word only
        pattern = r"\b" + re.escape(lower_word) + r"\b"
        matches = re.findall(pattern, lower_response)
        is_correct = len(matches) == expected_count
        confidence = 1.0  # High confidence in word count
        return (
            is_correct,
            confidence,
            {
                "word": word,
                "expected_count": expected_count,
                "actual_count": len(matches),
            },
        )

    def _check_numbered_list(
        self, response: str, expected_items: int
    ) -> tuple[bool, float, dict]:
        """Check if response contains a numbered list with expected items."""
        # Look for numbered items like "1.", "2.", etc.
        numbered_items = re.findall(r"^\s*\d+\.", response, re.MULTILINE)
        is_correct = len(numbered_items) == expected_items
        confidence = 0.9
        return (
            is_correct,
            confidence,
            {
                "expected_items": expected_items,
                "actual_items": len(numbered_items),
                "items": numbered_items,
            },
        )
