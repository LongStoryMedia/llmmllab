import re
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod


class AnswerExtractor(ABC):
    """Base class for deterministic answer extraction."""

    @abstractmethod
    def extract(
        self, response: str, question_data: Dict[str, Any]
    ) -> Tuple[str, float]:
        """
        Extract answer from response.

        Returns:
            Tuple of (extracted_answer, confidence_score)
            confidence_score: 0.0-1.0, where 1.0 means highly confident in extraction
        """


class MultipleChoiceExtractor(AnswerExtractor):
    """Deterministic multiple choice answer extractor."""

    def __init__(self, valid_choices: List[str] = ["A", "B", "C", "D"]):
        self.valid_choices = [choice.upper() for choice in valid_choices]

        # Ordered patterns from most to least reliable
        self.extraction_patterns = [
            # Pattern 1: Repeated letters (like "AA", "BB", etc.) - very high confidence
            (r"^([ABCD])\1+\s*$", 0.98),  # Matches AA, BB, CC, DD at start of line
            (
                r"([ABCD])\1+(?:\s|$)",
                0.95,
            ),  # Matches AA, BB anywhere with word boundary
            # Pattern 2: Explicit answer formats (highest confidence)
            (r"(?:final answer|answer|my answer)(?:\s+is)?:?\s*([ABCD])", 0.95),
            (r"(?:the answer is|answer is):\s*([ABCD])", 0.95),
            (r"(?:i choose|i select|selecting|choosing):?\s*([ABCD])", 0.90),
            # Pattern 3: Formatted responses (high confidence)
            (r"^\s*([ABCD])\s*$", 0.85),  # Just the letter alone
            (r"^\s*([ABCD])[.)\s]", 0.80),  # Letter with punctuation at start
            # Pattern 4: After thinking tags or at end of response
            (r"</think>\s*([ABCD])", 0.90),  # After closing think tag
            (
                r"</?think>.*?([ABCD])(?:\s*$|\s*[.!]?\s*$)",
                0.85,
            ),  # Near end after think
            # Pattern 5: In context patterns (medium confidence)
            (r"option\s+([ABCD])", 0.75),
            (r"choice\s+([ABCD])", 0.75),
            (r"\(([ABCD])\)", 0.70),
            # Pattern 6: Last resort patterns (lower confidence)
            (r"([ABCD])[.)]?\s+(?:is|would be|correct)", 0.60),
            (r"answer.*?([ABCD])", 0.50),
        ]

    def extract(
        self, response: str, question_data: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Extract multiple choice answer with confidence scoring."""
        if not response or not response.strip():
            return "UNKNOWN", 0.0

        # Clean and prepare response
        response_clean = self._clean_response(response)

        # Try patterns in order of confidence
        for pattern, base_confidence in self.extraction_patterns:
            matches = list(
                re.finditer(pattern, response_clean, re.IGNORECASE | re.MULTILINE)
            )

            if matches:
                # Get the last match (often the final answer)
                last_match = matches[-1]
                extracted = last_match.group(1).upper()

                if extracted in self.valid_choices:
                    # Adjust confidence based on context
                    confidence = self._adjust_confidence(
                        base_confidence, response_clean, extracted, matches
                    )
                    return extracted, confidence

        # Special handling: Look for the very last single letter in the response
        # This catches cases where the model just outputs the letter at the end
        last_letter_match = re.search(
            r"([ABCD])(?:\s*$|\s*[.!]?\s*$)", response_clean, re.IGNORECASE
        )
        if last_letter_match:
            extracted = last_letter_match.group(1).upper()
            if extracted in self.valid_choices:
                return extracted, 0.75

        # Fallback: look for any valid choice mentioned
        choice_counts = {}
        for choice in self.valid_choices:
            count = len(re.findall(rf"\b{choice}\b", response_clean, re.IGNORECASE))
            if count > 0:
                choice_counts[choice] = count

        if choice_counts:
            # Return most frequently mentioned choice with low confidence
            most_common = max(choice_counts.items(), key=lambda x: x[1])
            return most_common[0], 0.3

        return "UNKNOWN", 0.0

    def _clean_response(self, response: str) -> str:
        """Clean response for better pattern matching."""
        # Remove excessive whitespace and normalize
        response = re.sub(r"\s+", " ", response.strip())

        # Handle thinking tags specially - extract content after closing tag
        think_pattern = r"<think>.*?</think>\s*(.*?)$"
        think_match = re.search(think_pattern, response, re.DOTALL | re.IGNORECASE)
        if think_match:
            # If there's content after </think>, prioritize that
            after_think = think_match.group(1).strip()
            if after_think:
                # Use both the content after think and the original response
                response = after_think + "\n" + response

        # Remove common thinking prefixes that might interfere
        thinking_patterns = [
            r"^(?:let me think|thinking|i think|considering|analyzing).*?(?:\.|:|\n)",
            r"^(?:step \d+|first|second|third|finally).*?(?:\.|:|\n)",
        ]

        for pattern in thinking_patterns:
            response = re.sub(pattern, "", response, flags=re.IGNORECASE | re.MULTILINE)

        return response.strip()

    def _adjust_confidence(
        self, base_confidence: float, response: str, extracted: str, matches: List
    ) -> float:
        """Adjust confidence based on context clues."""
        confidence = base_confidence

        # Boost confidence if answer appears multiple times consistently
        if len(matches) > 1 and all(m.group(1).upper() == extracted for m in matches):
            confidence = min(0.98, confidence + 0.1)

        # Boost confidence if it's a repeated letter pattern (like AA, BB)
        if re.search(rf"{extracted}{extracted}+", response, re.IGNORECASE):
            confidence = min(0.98, confidence + 0.1)

        # Reduce confidence if multiple different choices are mentioned
        other_choices = [c for c in self.valid_choices if c != extracted]
        other_mentions = sum(
            1
            for choice in other_choices
            if re.search(rf"\b{choice}\b", response, re.IGNORECASE)
        )

        if other_mentions > 2:
            confidence *= 0.8

        # Boost confidence for clear conclusion patterns
        conclusion_patterns = [
            r"(?:therefore|thus|so|hence|in conclusion).*?" + extracted,
            r"(?:final answer|my answer|the answer).*?" + extracted,
        ]

        for pattern in conclusion_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                confidence = min(0.98, confidence + 0.05)
                break

        # Boost confidence if answer appears at the very end of response
        if re.search(rf"{extracted}(?:\s*$|\s*[.!]?\s*$)", response, re.IGNORECASE):
            confidence = min(0.98, confidence + 0.05)

        return confidence


class MathAnswerExtractor(AnswerExtractor):
    """Deterministic mathematical answer extractor."""

    def __init__(self):
        self.answer_patterns = [
            # Most reliable patterns
            (r"(?:final answer|answer|solution)(?:\s+is)?:?\s*([^.\n]+)", 0.90),
            (r"(?:therefore|thus|so|hence),?\s*([^.\n]+)", 0.85),
            (r"(?:the answer is|answer is):?\s*([^.\n]+)", 0.85),
            # Boxed or emphasized answers
            (r"\\boxed\{([^}]+)\}", 0.95),
            (r"\$([^$]+)\$(?:\s*(?:is the answer|$))", 0.80),
            # Equation solutions
            (r"([a-zA-Z]\s*=\s*[^,.\n]+)", 0.75),
            (r"=\s*([0-9.-]+(?:\s*or\s*[0-9.-]+)?)", 0.70),
        ]

    def extract(
        self, response: str, question_data: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Extract mathematical answer with confidence scoring."""
        if not response or not response.strip():
            return "UNKNOWN", 0.0

        response_clean = self._clean_math_response(response)
        expected_answer = question_data.get("answer", "").strip()

        # Try extraction patterns
        for pattern, base_confidence in self.answer_patterns:
            matches = list(
                re.finditer(pattern, response_clean, re.IGNORECASE | re.MULTILINE)
            )

            if matches:
                # Get the last match (final answer)
                last_match = matches[-1]
                extracted = last_match.group(1).strip()

                # Clean the extracted answer
                extracted = self._normalize_math_answer(extracted)

                if extracted and extracted != "UNKNOWN":
                    confidence = self._calculate_math_confidence(
                        extracted, expected_answer, base_confidence, response_clean
                    )
                    return extracted, confidence

        # Fallback: look for numbers or expressions at the end
        end_patterns = [
            r"([0-9.-]+(?:\.[0-9]+)?)(?:\s*$|\s*[.!]?\s*$)",
            r"([a-zA-Z]\s*=\s*[0-9.-]+)(?:\s*$|\s*[.!]?\s*$)",
        ]

        for pattern in end_patterns:
            match = re.search(pattern, response_clean)
            if match:
                extracted = self._normalize_math_answer(match.group(1))
                if extracted:
                    return extracted, 0.4

        return "UNKNOWN", 0.0

    def _clean_math_response(self, response: str) -> str:
        """Clean mathematical response for extraction."""
        # Remove LaTeX commands but preserve content
        response = re.sub(r"\\[a-zA-Z]+\{([^}]+)\}", r"\1", response)
        response = re.sub(r"\$+", "", response)  # Remove $ symbols

        # Normalize spacing around operators
        response = re.sub(r"\s*([=+-])\s*", r" \1 ", response)

        return response.strip()

    def _normalize_math_answer(self, answer: str) -> str:
        """Normalize mathematical answer format."""
        if not answer:
            return "UNKNOWN"

        # Clean up the answer
        answer = answer.strip()
        answer = re.sub(r"\s+", " ", answer)

        # Remove trailing punctuation
        answer = re.sub(r"[.!,]+$", "", answer)

        # Normalize fractions and decimals
        answer = re.sub(r"(\d+)\s*/\s*(\d+)", r"\1/\2", answer)

        return answer

    def _calculate_math_confidence(
        self, extracted: str, expected: str, base_confidence: float, response: str
    ) -> float:
        """Calculate confidence for mathematical answer."""
        confidence = base_confidence

        # Exact match boosts confidence significantly
        if expected and self._math_answers_equivalent(extracted, expected):
            confidence = min(0.98, confidence + 0.15)

        # Check if answer appears in a clear concluding statement
        conclusion_indicators = [
            "therefore",
            "thus",
            "so",
            "hence",
            "final answer",
            "the answer is",
        ]

        answer_context = response.lower()
        if any(indicator in answer_context for indicator in conclusion_indicators):
            confidence = min(0.95, confidence + 0.05)

        return confidence

    def _math_answers_equivalent(self, answer1: str, answer2: str) -> bool:
        """Check if two mathematical answers are equivalent."""
        # Simple equivalence checking - can be expanded
        a1_clean = re.sub(r"\s+", "", answer1.lower())
        a2_clean = re.sub(r"\s+", "", answer2.lower())

        # Direct match
        if a1_clean == a2_clean:
            return True

        # Try to extract numbers and compare
        nums1 = re.findall(r"-?\d+\.?\d*", answer1)
        nums2 = re.findall(r"-?\d+\.?\d*", answer2)

        if nums1 and nums2:
            try:
                return abs(float(nums1[0]) - float(nums2[0])) < 1e-6
            except ValueError:
                pass

        return False


class CodeAnswerExtractor(AnswerExtractor):
    """Deterministic code answer extractor."""

    def extract(
        self, response: str, question_data: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Extract code with confidence scoring."""
        if not response:
            return "UNKNOWN", 0.0

        # Look for code blocks
        code_block_patterns = [
            (r"```python\n(.*?)\n```", 0.90),
            (r"```\n(.*?)\n```", 0.85),
            (r"```(.*?)```", 0.80),
        ]

        for pattern, confidence in code_block_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                code = match.group(1).strip()
                if self._validate_code_structure(code, question_data):
                    return code, confidence

        # Fallback: look for function definitions
        func_pattern = r"(def\s+\w+.*?)(?=\ndef|\Z)"
        match = re.search(func_pattern, response, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if self._validate_code_structure(code, question_data):
                return code, 0.60

        return "UNKNOWN", 0.0

    def _validate_code_structure(
        self, code: str, question_data: Dict[str, Any]
    ) -> bool:
        """Validate that extracted code has proper structure."""
        if not code:
            return False

        # Must have function definition
        if "def " not in code:
            return False

        # Must have return statement for most problems
        if "return " not in code and question_data.get("requires_return", True):
            return False

        # Must be longer than the original prompt
        original_prompt = question_data.get("prompt", "")
        if len(code) <= len(original_prompt) * 1.2:
            return False

        return True
