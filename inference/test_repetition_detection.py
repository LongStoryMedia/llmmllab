#!/usr/bin/env python3
"""Test script for the updated repetition detection system."""

import re


def _detect_semantic_repetition_conservative(text: str) -> bool:
    """Ultra-conservative semantic repetition detection for very obvious cases only."""
    if len(text) < 600:  # Increased from 400 to be more conservative
        return False

    # Only look at the very end of the text to catch active loops
    recent_text = text[-300:]  # Only check last 300 characters

    # Split into sentences and check for near-identical content
    sentences = re.split(r"[.!?]+", recent_text)
    sentences = [
        s.strip() for s in sentences if len(s.strip()) > 30
    ]  # Increased from 20

    if len(sentences) < 3:  # Reduced from 4 - need fewer sentences
        return False

    # Only check for very high similarity (near-exact duplicates)
    recent_sentences = sentences[-3:]  # Only check last 3 sentences
    for i, sent1 in enumerate(recent_sentences):
        for sent2 in recent_sentences[i + 1 :]:
            words1 = set(re.findall(r"\b\w+\b", sent1.lower()))
            words2 = set(re.findall(r"\b\w+\b", sent2.lower()))

            # Both sentences must be substantial and very similar
            if len(words1) > 8 and len(words2) > 8:  # Increased from 5
                overlap = len(words1 & words2) / min(len(words1), len(words2))
                if (
                    overlap > 0.95
                ):  # Increased from 0.85 - only catch near-exact duplicates
                    return True

    return False


def _is_structured_content(text: str) -> bool:
    """Check if text contains structured content that should be exempt from repetition detection."""
    # JSON detection with generalized patterns
    json_patterns = [
        r'"\w+"\s*:\s*[^,}]+',  # Key-value pairs
        r'\{[^}]*"\w+"[^}]*\}',  # JSON objects
        r'"\w+"\s*:\s*\{[^}]*\}',  # Nested objects
    ]

    # Table detection patterns
    table_patterns = [
        r"\|.*\|.*\|",  # Markdown tables
        r"\s*[+|-]+\s*",  # ASCII table borders
    ]

    # Code block detection
    code_patterns = [
        r"```[\s\S]*?```",  # Markdown code blocks
        r"`[^`]+`",  # Inline code
        r"def \w+\(",  # Function definitions
        r"class \w+[\(:]?",  # Class definitions
    ]

    all_patterns = json_patterns + table_patterns + code_patterns

    for pattern in all_patterns:
        if re.search(pattern, text, re.MULTILINE):
            return True

    return False


def main():
    print("Testing updated repetition detection...")

    # Test 1: Technical content that should NOT trigger
    tech_text = """This is a technical document about API configuration and system parameters. The configuration specifies parameters for API endpoints and authentication protocols. The API parameters include authentication details and endpoint specifications for secure access. Configuration files define the API behavior and authentication requirements for proper system operation. More configuration details follow with additional parameters and security considerations for the API endpoints and authentication mechanisms."""

    result1 = _detect_semantic_repetition_conservative(tech_text)
    print(f"Technical content detection (should be False): {result1}")

    # Test 2: JSON content should be detected as structured
    json_text = """Here is JSON configuration:
    {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "mydb"
        },
        "cache": {
            "host": "redis-server",
            "port": 6379
        }
    }"""

    result2 = _is_structured_content(json_text)
    print(f"JSON detection (should be True): {result2}")

    # Test 3: Actually repetitive content should trigger (with enough repetition)
    repetitive_text = """The server failed to connect to the database immediately. The server failed to connect to the database immediately. The server failed to connect to the database immediately and continued failing. The server failed to connect to the database immediately and continued failing. The server failed to connect to the database immediately and continued failing repeatedly. The server failed to connect to the database immediately and continued failing repeatedly with the same error message."""

    result3 = _detect_semantic_repetition_conservative(repetitive_text)
    print(f"Repetitive content detection (should be True): {result3}")

    # Test 4: Table content should be detected as structured
    table_text = """Here is a comparison table:
    | Feature | Value | Description |
    |---------|-------|-------------|
    | CPU     | 4     | Cores       |
    | RAM     | 8GB   | Memory      |"""

    result4 = _is_structured_content(table_text)
    print(f"Table detection (should be True): {result4}")

    print("All tests completed!")

    # Summary
    success_count = sum(
        [
            not result1,  # Technical content should be False
            result2,  # JSON should be True
            result3,  # Repetitive should be True
            result4,  # Table should be True
        ]
    )

    print(f"Tests passed: {success_count}/4")
    if success_count == 4:
        print("✅ All repetition detection improvements working correctly!")
    else:
        print("❌ Some tests failed - review detection logic")


if __name__ == "__main__":
    main()
