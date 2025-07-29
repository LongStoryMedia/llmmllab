from typing import List, Dict, Any, Optional
import random
from datasets import load_dataset
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
        self.huggingface_dataset = None  # Will be loaded when needed

    def load_huggingface_dataset(self, subjects: Optional[List[str]] = None) -> None:
        """Load MMLU dataset from Hugging Face.

        Args:
            subjects: Optional list of subjects to filter. If None, all subjects are loaded.
                     Available subjects include: abstract_algebra, anatomy, astronomy, business_ethics, etc.
        """
        if self.huggingface_dataset is not None:
            return  # Already loaded

        try:
            # Load the dev split as default (can also use 'test' or 'validation')
            dataset = load_dataset("cais/mmlu", split="dev")

            # Convert to list to avoid Dataset API issues
            data_list = []
            for i in range(len(dataset)):
                data_list.append(dataset[i])

            # Filter by subjects if specified
            if subjects:
                data_list = [item for item in data_list if item["subject"] in subjects]

            # Store the list of dictionaries
            self.huggingface_dataset = data_list

            self.logger.info(
                f"Loaded {len(self.huggingface_dataset)} questions from MMLU dataset"
            )
            self.logger.info(
                f"Available subjects: {set(item['subject'] for item in self.huggingface_dataset)}"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to load MMLU dataset from Hugging Face: {str(e)}"
            )
            self.logger.warning("Falling back to sample questions")
            self.huggingface_dataset = None

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

    def get_questions_from_huggingface(
        self, num_questions: int, subjects: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get questions from the Hugging Face MMLU dataset.

        Args:
            num_questions: Number of questions to retrieve
            subjects: Optional list of subjects to filter

        Returns:
            List of questions in the format expected by the benchmark
        """
        # Load the dataset if not already loaded
        if self.huggingface_dataset is None:
            self.load_huggingface_dataset(subjects)

        # If loading failed, fall back to sample questions
        if self.huggingface_dataset is None:
            print(
                "Failed to load Hugging Face dataset, falling back to sample questions"
            )
            return self.get_sample_questions()

        # Get dataset info
        print(f"Loaded Hugging Face MMLU dataset successfully")

        try:
            # Print dataset statistics
            all_subjects = set(item["subject"] for item in self.huggingface_dataset)
            print(f"Available subjects in dataset: {len(all_subjects)}")
            print(f"Some example subjects: {list(all_subjects)[:5]}")
            dataset_size = len(self.huggingface_dataset)
            print(f"Total questions available: {dataset_size}")

            # Randomly sample questions
            if num_questions >= dataset_size:
                # If we want more questions than available, use all and duplicate if needed
                indices = list(range(dataset_size))
                if num_questions > dataset_size:
                    indices = indices * (num_questions // dataset_size + 1)
                indices = indices[:num_questions]
                print(f"Using all {dataset_size} questions (with some duplication)")
            else:
                # Otherwise, randomly sample without replacement
                indices = random.sample(range(dataset_size), num_questions)
                print(f"Randomly sampled {num_questions} questions from dataset")

            # Convert to the format expected by our benchmark
            questions = []
            for idx in indices:
                example = self.huggingface_dataset[idx]
                question = {
                    "subject": example["subject"],
                    "question": example["question"],
                    "choices": [example["A"], example["B"], example["C"], example["D"]],
                    "answer": example[
                        "answer"
                    ],  # This is already in "A", "B", "C", "D" format
                }
                questions.append(question)

            # Print subject distribution
            subject_counts = {}
            for q in questions:
                subject_counts[q["subject"]] = subject_counts.get(q["subject"], 0) + 1
            print(f"Questions by subject:")
            for subject, count in sorted(
                subject_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]:
                print(f"  {subject}: {count}")

            return questions
        except Exception as e:
            print(f"Error processing Hugging Face dataset: {str(e)}")
            return self.get_sample_questions()

    def run(
        self,
        model_id: str,
        num_samples: int = 100,
        use_huggingface: bool = True,
        subjects: Optional[List[str]] = None,
    ) -> BenchmarkResult:
        """Run MMLU benchmark on the specified model.

        Args:
            model_id: The ID of the model to evaluate
            num_samples: Number of questions to evaluate
            use_huggingface: Whether to use the Hugging Face MMLU dataset or sample questions
            subjects: Optional list of subjects to filter (only used if use_huggingface=True)
        """
        print("\n--- MMLU Benchmark ---")

        # Get questions from either Hugging Face dataset or sample questions
        if use_huggingface:
            try:
                questions = self.get_questions_from_huggingface(num_samples, subjects)
                num_samples = len(
                    questions
                )  # In case we couldn't get as many as requested
                self.logger.info(
                    f"Using {num_samples} questions from Hugging Face MMLU dataset"
                )
            except Exception as e:
                self.logger.error(f"Error loading Hugging Face dataset: {str(e)}")
                self.logger.info("Falling back to sample questions")
                questions = self.get_sample_questions()
                num_samples = self.validate_sample_count(num_samples)
                # Extend questions to match requested sample count
                questions = (questions * (num_samples // len(questions) + 1))[
                    :num_samples
                ]
        else:
            questions = self.get_sample_questions()
            num_samples = self.validate_sample_count(num_samples)
            # Extend questions to match requested sample count
            questions = (questions * (num_samples // len(questions) + 1))[:num_samples]

        correct_answers = 0
        detailed_results = []
        subject_scores = {}

        for i, question in enumerate(questions):
            print(f"MMLU Question {i+1}/{len(questions)}")

            # Format prompt following MMLU format
            # Use the multiple_choice_template from PromptTemplates
            prompt = PromptTemplates.multiple_choice_template(
                question=question["question"],
                choices=question["choices"],
                subject=question["subject"].replace("_", " "),
            )

            try:
                self._print_question_debug(question, i)

                response = self.inference_engine.run_single_inference(
                    model_id, prompt, max_tokens=10, temperature=0.0
                )
                full_response = response.get("response", "")

                print("\nMODEL RESPONSE:")
                print(f"{full_response}")

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

                # Track by subject
                subject = question["subject"]
                if subject not in subject_scores:
                    subject_scores[subject] = {"correct": 0, "total": 0}
                subject_scores[subject]["total"] += 1
                if is_correct:
                    subject_scores[subject]["correct"] += 1

                detailed_results.append(
                    {
                        "question_id": i,
                        "subject": subject,
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
                )

            except Exception as e:
                self.logger.error(f"Error in MMLU question {i}: {str(e)}")
                detailed_results.append(
                    {"question_id": i, "subject": question["subject"], "error": str(e)}
                )

        # Calculate subject-wise scores
        subject_accuracy = {}
        for subject, scores in subject_scores.items():
            subject_accuracy[subject] = (
                scores["correct"] / scores["total"] if scores["total"] > 0 else 0
            )

        mmlu_score = correct_answers / len(questions) if questions else 0

        print(f"MMLU Score: {mmlu_score:.3f} ({correct_answers}/{len(questions)})")

        return BenchmarkResult(
            score=mmlu_score,
            total_questions=len(questions),
            correct_answers=correct_answers,
            detailed_results=detailed_results[:10],  # Limit detailed results
            metadata={"subject_accuracy": subject_accuracy},
        )

    def _print_question_debug(self, question: Dict, index: int) -> None:
        """Print debug info for a question."""
        print("\n" + "-" * 50)
        print(f"SUBJECT: {question['subject']}")
        print(f"QUESTION: {question['question']}")
        print("CHOICES:")
        for i, choice in enumerate(question["choices"]):
            print(f"{chr(65+i)}. {choice}")
        print(f"CORRECT ANSWER: {question['answer']}")
        print("-" * 50 + "\n")
