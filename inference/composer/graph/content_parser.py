"""Content parsing utilities for workflow executor."""

import re
from typing import List, Tuple


# Detect raw tool-call XML that the model sometimes emits inline in content
# when it generates text before a tool call. llama.cpp fails to parse the
# tool portion as structured, so the whole thing arrives as content text.
# Handles <tool_call>, <function_call>, and <|tool_call|> variants, with
# possible whitespace / newlines between < and the tag name.
_RAW_TOOL_CALL_RE = re.compile(
    r"<\s*\|?\s*(?:tool_call|function_call|tool-call|function-call)\s*\|?\s*>",
    re.IGNORECASE | re.DOTALL,
)
# Match complete tool-call blocks (or unclosed at EOF).
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<\s*\|?\s*(?:tool_call|function_call|tool[-_]call|function[-_]call)\s*\|?\s*>"
    r"(.*?)"
    r"(?:<\s*/\s*\|?\s*(?:tool_call|function_call|tool[-_]call|function[-_]call)\s*\|?\s*>|$)",
    re.IGNORECASE | re.DOTALL,
)


def strip_think_tags(text: str, think_closed: bool = False) -> Tuple[str, str, bool]:
    """
    Split text on </think> boundary.

    Args:
        text: Text that may contain <think> tags
        think_closed: Whether thinking section is already closed

    Returns:
        Tuple of (thinking_part, content_part, new_think_closed)
        - If </think> found: returns (thinking content, rest after tag, True)
        - If no </think> and not closed yet: returns (text, "", False)
        - If no </think> and already closed: returns ("", text, False)
    """
    if "</think>" in text:
        before, after = text.split("</think>", 1)
        # Strip <think> prefix if present
        before = before.lstrip()
        if before.startswith("<think>"):
            before = before[len("<think>") :]
        return before.strip(), after.lstrip("\n"), True
    if not think_closed:
        # Haven't seen </think> yet; buffer as thinking
        return text, "", False
    return "", text, False


def parse_content(content: str | List[str | dict]) -> List[str]:
    """
    Parse message content into a list of strings.

    Args:
        content: Content which can be a string or list of strings/dicts

    Returns:
        List[str]: Parsed list of string content
    """
    if isinstance(content, str):
        return [content]
    else:
        return [str(c) for c in content]
