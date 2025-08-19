from typing import List, Dict, Any
from ..base.benchmark_base import BenchmarkBase
from ..base.result_types import BenchmarkResult
from ..utils.inference import InferenceEngine
from ..utils.deterministic_extractors import MultipleChoiceExtractor
from ..utils.deterministic_evaluators import MultipleChoiceEvaluator
from ..utils.prompt_templates import PromptTemplates


class GPQADiamondBenchmark(BenchmarkBase):
    """GPQA-Diamond benchmark - Graduate-level science questions."""

    def __init__(self):
        super().__init__(
            name="GPQA-Diamond",
            description="Graduate-level Physics, Chemistry, Biology Q&A",
        )
        self.inference_engine = InferenceEngine()
        self.extractor = MultipleChoiceExtractor()
        self.evaluator = MultipleChoiceEvaluator()

    def get_sample_questions(self) -> List[Dict[str, Any]]:
        """Get sample GPQA-Diamond style questions."""
        return [
            {
                "subject": "Chemistry",
                "question": "Consider a circular nanostructure. Quantum confinement along its circumference would lead to which of the following electron behavior?",
                "choices": [
                    "Only energy states with integer spin quantum numbers are allowed",
                    "The electron wavelength must fit an integer number of times into the circumference",
                    "A magnetic field is induced perpendicular to the plane of the circle",
                    "There is no quantum confinement along a circle",
                ],
                "answer": "B",
            },
            {
                "subject": "Biology",
                "question": "Which of the following most accurately describes the function of CpG islands in the human genome?",
                "choices": [
                    "They are regions of alternative splicing for RNA transcripts",
                    "They serve as recognition sites for DNA methylation and gene silencing",
                    "They are promoter regions often associated with constitutively expressed genes",
                    "They encode non-coding RNAs that regulate gene expression",
                ],
                "answer": "C",
            },
            {
                "subject": "Physics",
                "question": "In the WKB approximation for quantum tunneling through a barrier, the transmission probability depends exponentially on which of the following quantities?",
                "choices": [
                    "The square root of the barrier height",
                    "The square of the barrier width",
                    "The integral of the square root of the potential minus the energy across the barrier width",
                    "The ratio of the particle's mass to the barrier height",
                ],
                "answer": "C",
            },
            {
                "subject": "Medicine",
                "question": "A 65-year-old male on long-term ACE inhibitor therapy presents with angioedema. Which mediator is most directly involved in the pathophysiology of this adverse effect?",
                "choices": [
                    "Histamine",
                    "Bradykinin",
                    "Leukotrienes",
                    "Prostaglandins",
                ],
                "answer": "B",
            },
        ]

    def run(self, model_id: str, num_samples: int = 10) -> BenchmarkResult:
        """Run the GPQA-Diamond benchmark on specified model."""
        print("\n--- GPQA-Diamond Benchmark ---")

        questions = self.get_sample_questions()
        num_samples = self.validate_sample_count(num_samples, max_samples=10)

        # Limit samples to available questions
        extended_questions = questions[:num_samples]

        correct_answers = 0
        detailed_results = []

        for i, question in enumerate(extended_questions):
            print(f"GPQA Question {i+1}/{len(extended_questions)}")

            # Use the multiple_choice_template from PromptTemplates
            prompt = PromptTemplates.multiple_choice_template(
                question=question["question"],
                choices=question["choices"],
                subject=question["subject"],
            )

            try:
                self._print_question_debug(question)

                response = self.inference_engine.run_single_inference(
                    model_id, prompt, max_tokens=200, temperature=0.1
                )
                full_response = response.get("response", "")

                print("\nMODEL RESPONSE:")
                print(
                    f"{full_response[:500]}{'...' if len(full_response) > 500 else ''}"
                )

                extracted_answer, confidence = self.extractor.extract(
                    full_response, question
                )

                print("\nEXTRACTION:")
                print(
                    f"Extracted answer: {extracted_answer} (confidence: {confidence:.2f})"
                )

                is_correct, eval_confidence, metadata = self.evaluator.evaluate(
                    extracted_answer, question["answer"], question, confidence
                )
                print(f"Correct? {'YES' if is_correct else 'NO'}")

                if is_correct:
                    correct_answers += 1

                detailed_results.append(
                    {
                        "question_id": i,
                        "subject": question["subject"],
                        "correct_answer": question["answer"],
                        "model_answer": extracted_answer,
                        "extraction_confidence": confidence,
                        "eval_confidence": eval_confidence,
                        "is_correct": is_correct,
                        "response": (
                            full_response[:100] + "..."
                            if len(full_response) > 100
                            else full_response
                        ),
                    }
                )

            except Exception as e:
                self.logger.error(f"Error in GPQA question {i}: {str(e)}")
                detailed_results.append(
                    {"question_id": i, "subject": question["subject"], "error": str(e)}
                )

        gpqa_score = (
            correct_answers / len(extended_questions) if extended_questions else 0
        )

        print(
            f"GPQA-Diamond Score: {gpqa_score:.3f} ({correct_answers}/{len(extended_questions)})"
        )

        return BenchmarkResult(
            score=gpqa_score,
            total_questions=len(extended_questions),
            correct_answers=correct_answers,
            detailed_results=detailed_results,
            metadata={
                "benchmark": "GPQA-Diamond",
                "description": "Graduate-level science questions",
            },
        )

    def _print_question_debug(self, question: Dict) -> None:
        """Print debug info for a question."""
        print("\n" + "-" * 50)
        print(f"SUBJECT: {question['subject']}")
        print(f"QUESTION: {question['question']}")
        print("CHOICES:")
        for i, choice in enumerate(question["choices"]):
            print(f"{chr(65+i)}. {choice}")
        print(f"CORRECT ANSWER: {question['answer']}")
        print("-" * 50 + "\n")
