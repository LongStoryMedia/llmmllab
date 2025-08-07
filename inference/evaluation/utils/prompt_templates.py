from typing import List, Optional, Dict, Any


class PromptTemplates:
    """Standardized prompt templates for deterministic evaluation."""

    @staticmethod
    def multiple_choice_template(
        question: str, choices: List[str], subject: Optional[str] = None
    ) -> str:
        """Template for multiple choice questions."""
        subject_prefix = f"This is a {subject} question.\n\n" if subject else ""

        choices_text = "\n".join(
            [f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]
        )

        return f"""{subject_prefix}Question: {question}

{choices_text}

Instructions: Select the single best answer. Respond with only the letter (A, B, C, or D) of your choice.

Answer:"""

    @staticmethod
    def math_template(problem: str, show_work: bool = False) -> str:
        """Template for mathematical problems."""
        work_instruction = (
            "Show your work step by step, then provide your final answer clearly."
            if show_work
            else "Provide your final answer clearly."
        )

        return f"""Problem: {problem}

Instructions: {work_instruction}

Solution:"""

    @staticmethod
    def code_template(prompt: str, function_name: Optional[str] = None) -> str:
        """Template for code generation problems."""
        name_hint = f" (function name: {function_name})" if function_name else ""

        return f"""Complete the following Python function{name_hint}:

{prompt}

Instructions: Provide only the complete function implementation. Ensure your code is syntactically correct and handles the specified requirements.

```python"""

    @staticmethod
    def instruction_following_template(instruction: str) -> str:
        """Template for instruction following tasks."""
        return f"""Task: {instruction}

Instructions: Follow the given task exactly as specified. Pay attention to all requirements and constraints.

Response:"""

    @staticmethod
    def livebench_template(question: str, category: str) -> str:
        """Template for LiveBench questions of varying types."""
        category_prompt = {
            "reasoning": "Think through this step by step:",
            "knowledge": "Provide an up-to-date, accurate response:",
            "instruction_following": "Follow these instructions precisely:",
        }.get(category, "Please answer the following question:")

        return f"""{category_prompt}

{question}

Answer:"""
