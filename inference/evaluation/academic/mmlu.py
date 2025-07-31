from typing import List, Dict, Any, Optional, Tuple
import random
from ..base.benchmark_base import BenchmarkBase
from ..base.result_types import BenchmarkResult
from ..utils.inference import InferenceEngine
from ..utils.deterministic_extractors import MultipleChoiceExtractor
from ..utils.deterministic_evaluators import MultipleChoiceEvaluator
from ..utils.prompt_templates import PromptTemplates


class MMLUBenchmark(BenchmarkBase):
    """
    MMLU (Massive Multitask Language Understanding) benchmark.

    MMLU consists of 15,908 multiple-choice questions across 57 subjects.
    Format: Question with 4 options (A, B, C, D), expecting single letter answer.
    """

    def __init__(self):
        super().__init__(
            name="MMLU", description="Massive Multitask Language Understanding"
        )
        self.inference_engine = InferenceEngine()
        self.extractor = MultipleChoiceExtractor()
        self.evaluator = MultipleChoiceEvaluator()

    def get_sample_questions(self) -> List[Dict[str, Any]]:
        """Get sample MMLU questions across different domains."""
        return [
            {
                "subject": "abstract_algebra",
                "question": "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.",
                "choices": ["0", "4", "2", "6"],
                "answer": "B",
            },
            {
                "subject": "anatomy",
                "question": "A patient suffers a broken nose, you would record this injury as involving which bone?",
                "choices": ["Vomer", "Nasal", "Maxilla", "Mandible"],
                "answer": "B",
            },
            {
                "subject": "astronomy",
                "question": "Where do most short-period comets come from and how do we know?",
                "choices": [
                    "The Kuiper belt; short period comets tend to be in the plane of the solar system just like the Kuiper belt.",
                    "The Kuiper belt; short period comets tend to come from random directions indicating a spherical distribution of comets called the Kuiper belt.",
                    "The Oort cloud; short period comets tend to be in the plane of the solar system just like the Oort cloud.",
                    "The Oort cloud; short period comets tend to come from random directions indicating a spherical distribution of comets called the Oort cloud.",
                ],
                "answer": "A",
            },
            {
                "subject": "business_ethics",
                "question": "What is the judge's standard of review in business judgment rule?",
                "choices": [
                    "Substituted judgment standard",
                    "Best interest standard",
                    "Rational basis review",
                    "Gross negligence standard",
                ],
                "answer": "C",
            },
            {
                "subject": "clinical_knowledge",
                "question": "A 65-year-old male smoker presents with a 3-month history of weight loss and right upper quadrant pain. Which investigation is most appropriate as the first line?",
                "choices": [
                    "ERCP",
                    "Abdominal ultrasound",
                    "CT scan of chest",
                    "Upper GI endoscopy",
                ],
                "answer": "B",
            },
            {
                "subject": "college_biology",
                "question": "Which of the following is not a way to form recombinant DNA?",
                "choices": [
                    "Translation",
                    "Conjugation",
                    "Specialized transduction",
                    "Transformation",
                ],
                "answer": "A",
            },
            {
                "subject": "college_chemistry",
                "question": "3 Cl−(aq) + 4 CrO_4^2−(aq) + 23 H+(aq) → 3 HClO2(aq) + 4 Cr^3+(aq) + 10 H2O(l). In the reaction shown above, Cl−(aq) behaves as",
                "choices": ["an acid", "a base", "a catalyst", "a reducing agent"],
                "answer": "D",
            },
            {
                "subject": "computer_security",
                "question": "SHA-1 has a message digest of",
                "choices": ["160 bits", "512 bits", "628 bits", "820 bits"],
                "answer": "A",
            },
            {
                "subject": "conceptual_physics",
                "question": "Colors in a soap bubble result from light",
                "choices": [
                    "converted to a different frequency",
                    "deflection",
                    "interference",
                    "polarization",
                ],
                "answer": "C",
            },
            {
                "subject": "econometrics",
                "question": "For a stationary AR(1) process, shocks will",
                "choices": [
                    "Eventually die away",
                    "Persist indefinitely",
                    "Grow exponentially",
                    "Never occur",
                ],
                "answer": "A",
            },
        ]

    def run(
        self,
        model_id: str,
        num_samples: int = 100,
        dataset_path: Optional[str] = None,
        subjects: Optional[List[str]] = None,
    ) -> BenchmarkResult:
        """Run MMLU benchmark on the specified model.

        Args:
            model_id: The ID of the model to evaluate
            num_samples: Number of questions to evaluate
            dataset_path: Optional path to a HuggingFace dataset in the format "space/dataset-name"
            subjects: Optional list of subjects to filter
        """
        print("\n--- MMLU Benchmark ---")
        total_answers = 0
        correct_answers = 0
        detailed_results = []
        subject_scores = {}

        def aq(q):
            is_correct, result = self.answer_question(model_id, q)
            detailed_results.append(result)
            nonlocal total_answers
            nonlocal correct_answers
            total_answers += 1
            if is_correct:
                correct_answers += 1

            # Track subject scores if subjects are specified
            if subjects and q["subject"] in subjects:
                if q["subject"] not in subject_scores:
                    subject_scores[q["subject"]] = {"correct": 0, "total": 0}
                subject_scores[q["subject"]]["total"] += 1
                if is_correct:
                    subject_scores[q["subject"]]["correct"] += 1
            elif not subjects:  # Track all subjects if none specified
                if q["subject"] not in subject_scores:
                    subject_scores[q["subject"]] = {"correct": 0, "total": 0}
                subject_scores[q["subject"]]["total"] += 1
                if is_correct:
                    subject_scores[q["subject"]]["correct"] += 1

        questions_processed = False

        # Try to get questions from HuggingFace dataset first
        if dataset_path:
            try:
                # Load dataset with proper error handling
                dataset = self.load_dataset_from_huggingface(dataset_path)

                if dataset and hasattr(dataset, "__iter__"):
                    self.logger.info(f"Successfully loaded dataset: {dataset_path}")
                    c = 0

                    for q in dataset:
                        if c >= num_samples:
                            break

                        # Skip if subjects filter is specified and question doesn't match
                        if (
                            subjects
                            and isinstance(q, Dict)
                            and q.get("subject", "") not in subjects
                        ):
                            continue

                        print(f"MMLU Question {c+1}/{num_samples}")
                        aq(q)
                        c += 1

                    questions_processed = True

                else:
                    self.logger.warning(
                        "Dataset is empty or invalid, falling back to sample questions"
                    )

            except Exception as e:
                self.logger.error(f"Error loading HuggingFace dataset: {str(e)}")
                self.logger.info("Falling back to sample questions")

        # Fall back to sample questions if dataset loading failed or no dataset specified
        if not questions_processed:
            questions = self.get_sample_questions()

            # Filter by subjects if specified
            if subjects:
                questions = [q for q in questions if q["subject"] in subjects]

            if not questions:
                self.logger.error("No questions available after filtering")
                return BenchmarkResult(
                    score=0.0,
                    total_questions=0,
                    correct_answers=0,
                    detailed_results=[],
                    metadata={"error": "No questions available after filtering"},
                )

            # Adjust num_samples and extend questions if needed
            num_samples = min(
                num_samples, len(questions) * 10
            )  # Allow cycling through questions

            # Extend questions to match requested sample count by cycling
            if num_samples > len(questions):
                questions = (questions * (num_samples // len(questions) + 1))[
                    :num_samples
                ]
            else:
                questions = questions[:num_samples]

            for i, question in enumerate(questions):
                print(f"MMLU Question {i+1}/{len(questions)}")
                aq(question)

        # Calculate subject-wise scores
        subject_accuracy = {}
        for subject, scores in subject_scores.items():
            subject_accuracy[subject] = (
                scores["correct"] / scores["total"] if scores["total"] > 0 else 0
            )

        # Avoid division by zero
        if total_answers == 0:
            self.logger.error("No questions were processed")
            return BenchmarkResult(
                score=0.0,
                total_questions=0,
                correct_answers=0,
                detailed_results=[],
                metadata={"error": "No questions were processed"},
            )

        mmlu_score = correct_answers / total_answers

        print(f"MMLU Score: {mmlu_score:.3f} ({correct_answers}/{total_answers})")

        return BenchmarkResult(
            score=mmlu_score,
            total_questions=total_answers,
            correct_answers=correct_answers,
            detailed_results=detailed_results[:10],  # Limit detailed results
            metadata={"subject_accuracy": subject_accuracy},
        )

    def answer_question(
        self, model_id: str, question: Dict
    ) -> Tuple[bool, Dict]:  # Format prompt following MMLU format
        # Use the multiple_choice_template from PromptTemplates
        prompt = PromptTemplates.multiple_choice_template(
            question=question["question"],
            choices=question["choices"],
            subject=question["subject"].replace("_", " "),
        )

        self._print_question_debug(question)
        response = self.inference_engine.run_single_inference(
            model_id=model_id, prompt=prompt
        )
        full_response = response.get("response", "")
        print("\nMODEL RESPONSE:")
        print(f"{full_response}")
        extracted_answer, confidence = self.extractor.extract(full_response, question)
        print("\nEXTRACTION:")
        print(f"Extracted answer: {extracted_answer} (confidence: {confidence:.2f})")
        is_correct, eval_confidence, _ = self.evaluator.evaluate(
            extracted_answer, question["answer"], question, confidence
        )
        print(f"Correct? {'YES' if is_correct else 'NO'}")
        return is_correct, {
            "subject": question["subject"],
            "question": (
                question["question"][:100] + "..."
                if len(question["question"]) > 100
                else question["question"]
            ),
            "correct_answer": question["answer"],
            "model_answer": extracted_answer,
            "extraction_confidence": confidence,
            "eval_confidence": eval_confidence,
            "is_correct": is_correct,
            "response": full_response[:50],
        }

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
