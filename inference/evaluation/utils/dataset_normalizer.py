from typing import Dict, List, Any


def normalize_mmlu_like(example: Dict[str, Any]) -> Dict[str, Any]:
    """Standardize MMLU-like examples into: {question, choices, answer, subject}"""
    if "choices" in example and isinstance(example["choices"], list):
        return {
            "question": example["question"],
            "choices": example["choices"],
            "answer": example["answer"],
            "subject": example.get("subject", "unknown"),
        }

    if all(k in example for k in ["option_a", "option_b", "option_c", "option_d"]):
        choices = [
            example["option_a"],
            example["option_b"],
            example["option_c"],
            example["option_d"],
        ]
        # Normalize answer: could be index or letter
        answer = example.get("answer")
        if isinstance(answer, int):
            answer = chr(65 + answer)  # A/B/C/D
        elif isinstance(answer, str) and answer.isdigit():
            answer = chr(65 + int(answer))

        return {
            "question": example["question"],
            "choices": choices,
            "answer": answer,
            "subject": example.get("subject", "unknown"),
        }

    raise ValueError("Unknown MMLU-like dataset format.")
