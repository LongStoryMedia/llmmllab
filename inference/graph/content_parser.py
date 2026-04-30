"""Content parsing utilities for workflow executor."""

import re
from typing import List, Tuple

# Detect the *opening* tag of a raw tool-call block that the model may emit
# inline in the content stream when llama.cpp fails to parse the tool portion
# as structured output.  Handles <tool_call>, <function_call>,
# <|tool_call|>, and hyphenated variants with optional whitespace.
_RAW_TOOL_CALL_RE = re.compile(
    r"<\s*\|?\s*(?:tool_call|function_call|tool-call|function-call)\s*\|?\s*>",
    re.IGNORECASE | re.DOTALL,
)

# Match the *closing* tag of a raw tool-call block (same variant set).
_RAW_TOOL_CALL_CLOSE_RE = re.compile(
    r"<\s*/\s*\|?\s*(?:tool_call|function_call|tool[-_]call|function[-_]call)\s*\|?\s*>",
    re.IGNORECASE | re.DOTALL,
)

# Match a *complete* tool-call block (open … close).  Used for batch parsing
# of already-buffered text where we know the full block is present.
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<\s*\|?\s*(?:tool_call|function_call|tool[-_]call|function[-_]call)\s*\|?\s*>"
    r"(.*?)"
    r"(?:<\s*/\s*\|?\s*(?:tool_call|function_call|tool[-_]call|function[-_]call)\s*\|?\s*>|$)",
    re.IGNORECASE | re.DOTALL,
)

# Detect Mistral-style [TOOL_CALLS] marker in streaming content.
_MISTRAL_TOOL_CALLS_RE = re.compile(r"\[TOOL_CALLS\]", re.IGNORECASE)

# Detect the start of a bare JSON tool call.  We look for the characteristic
# {"name": "...", "arguments"  pattern to avoid false positives on normal JSON.
_BARE_JSON_TOOL_CALL_RE = re.compile(
    r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"(?:arguments|args)"\s*:',
    re.DOTALL,
)


_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)


def clean_think_tags(text: str) -> str:
    """Remove all <think> and </think> tags from text."""
    return _THINK_TAG_RE.sub("", text).strip()


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
        - If no </think> and already closed: returns ("", text, True)
    """
    if "</think>" in text:
        before, after = text.split("</think>", 1)
        before = before.lstrip()
        if before.startswith("<think>"):
            before = before[len("<think>") :]
        return before.strip(), after.lstrip("\n"), True
    if not think_closed:
        return text, "", False
    return "", text, True


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
    result = []
    for c in content:
        if isinstance(c, dict) and "text" in c:
            text = c.get("text", "")
            if isinstance(text, str):
                result.append(text)
        else:
            result.append(str(c))
    return result


class RawToolCallStreamBuffer:
    """
    Accumulates streaming chunks so that raw tool-call XML blocks that arrive
    across multiple chunks are never partially emitted as visible text.

    Usage in a streaming loop::

        buf = RawToolCallStreamBuffer()
        for chunk_text in stream:
            safe_text, complete_blocks = buf.feed(chunk_text)
            # safe_text  → emit immediately as content_delta
            # complete_blocks → list of raw XML strings ready for parsing

        # After stream ends, flush any incomplete block (treat as tool call):
        leftover_text, leftover_blocks = buf.flush()

    The buffer guarantees that once a ``<tool_call>`` (or variant) opening tag
    is seen, *no* subsequent bytes are forwarded as ``safe_text`` until the
    matching closing tag is found.  This prevents partial XML from leaking to
    the client as garbled text.
    """

    def __init__(self) -> None:
        self._pending: str = ""  # accumulated text waiting for close tag / brace
        self._buffering: bool = False
        self._buffer_mode: str = "xml"  # "xml", "json", or "mistral"
        self._brace_depth: int = 0  # for JSON brace counting

    @property
    def is_buffering(self) -> bool:
        """True while we are inside a raw tool-call block."""
        return self._buffering

    def feed(self, text: str) -> Tuple[str, List[str]]:
        """
        Accept the next streaming chunk.

        Returns:
            (safe_text, complete_blocks)
            - safe_text: text that is safe to emit as a content delta right now
            - complete_blocks: zero or more complete raw tool-call strings
        """
        safe_prefix = ""
        if not self._buffering:
            # Not currently inside a raw tool call block.

            # 1. Check for XML-tagged tool calls.
            open_match = _RAW_TOOL_CALL_RE.search(text)
            if open_match is not None:
                safe_prefix = text[: open_match.start()]
                self._pending = text[open_match.start() :]
                self._buffering = True
                self._buffer_mode = "xml"
                # Fall through to buffering logic.
            else:
                # 2. Check for [TOOL_CALLS] marker.
                mistral_match = _MISTRAL_TOOL_CALLS_RE.search(text)
                if mistral_match is not None:
                    safe_prefix = text[: mistral_match.start()]
                    self._pending = text[mistral_match.start() :]
                    self._buffering = True
                    self._buffer_mode = "mistral"
                    # Mistral blocks end at double-newline or stream end.
                else:
                    # 3. Check for bare JSON tool call.
                    json_match = _BARE_JSON_TOOL_CALL_RE.search(text)
                    if json_match is not None:
                        safe_prefix = text[: json_match.start()]
                        self._pending = text[json_match.start() :]
                        self._buffering = True
                        self._buffer_mode = "json"
                        self._brace_depth = 0
                        for ch in self._pending:
                            if ch == "{":
                                self._brace_depth += 1
                            elif ch == "}":
                                self._brace_depth -= 1
                        # Fall through to check if JSON is already complete.
                    else:
                        # No tool-call markers at all — pass through.
                        return text, []

        else:
            # Already buffering — append new chunk.
            self._pending += text

        # --- Buffering logic: try to find a complete block ---
        complete_blocks: List[str] = []
        safe_text_parts: List[str] = []

        if self._buffer_mode == "xml":
            # Standard XML tool-call buffering.
            while self._buffering and self._pending:
                close_match = _RAW_TOOL_CALL_CLOSE_RE.search(self._pending)
                if close_match is None:
                    break

                block_end = close_match.end()
                complete_blocks.append(self._pending[:block_end])
                remainder = self._pending[block_end:]
                self._pending = ""
                self._buffering = False

                next_open = _RAW_TOOL_CALL_RE.search(remainder)
                if next_open is None:
                    safe_text_parts.append(remainder)
                else:
                    safe_text_parts.append(remainder[: next_open.start()])
                    self._pending = remainder[next_open.start() :]
                    self._buffering = True

        elif self._buffer_mode == "json":
            # Count braces to find complete JSON object(s).
            if not hasattr(self, "_brace_depth"):
                self._brace_depth = 0
            # Recount from scratch for accuracy.
            self._brace_depth = 0
            for i, ch in enumerate(self._pending):
                if ch == "{":
                    self._brace_depth += 1
                elif ch == "}":
                    self._brace_depth -= 1
                    if self._brace_depth == 0:
                        # Found complete JSON object.
                        block = self._pending[: i + 1]
                        complete_blocks.append(block)
                        remainder = self._pending[i + 1 :]
                        self._pending = ""
                        self._buffering = False
                        # Check if remainder has another JSON tool call.
                        next_json = _BARE_JSON_TOOL_CALL_RE.search(remainder)
                        if next_json is not None:
                            safe_text_parts.append(remainder[: next_json.start()])
                            self._pending = remainder[next_json.start() :]
                            self._buffering = True
                            self._buffer_mode = "json"
                            self._brace_depth = 0
                            for ch2 in self._pending:
                                if ch2 == "{":
                                    self._brace_depth += 1
                                elif ch2 == "}":
                                    self._brace_depth -= 1
                        else:
                            safe_text_parts.append(remainder)
                        break

        elif self._buffer_mode == "mistral":
            # Mistral: buffer until we find ]\n or stream ends.
            bracket_pos = self._pending.find("]\n")
            if bracket_pos == -1:
                bracket_pos = self._pending.rfind("]")
            if bracket_pos != -1 and bracket_pos > self._pending.find("["):
                block = self._pending[: bracket_pos + 1]
                complete_blocks.append(block)
                remainder = self._pending[bracket_pos + 1 :]
                self._pending = ""
                self._buffering = False
                safe_text_parts.append(remainder)

        final_safe = safe_prefix + "".join(safe_text_parts)
        return final_safe, complete_blocks

    def flush(self) -> Tuple[str, List[str]]:
        """
        Called when the model stream ends.

        If we are still buffering an incomplete block (the model stopped mid
        tool-call, which can happen with truncation), return it as a complete
        block anyway so the caller can attempt to parse it.  This prevents
        the partial XML from being silently dropped *or* leaked as text.

        Returns:
            (safe_text, complete_blocks) — same contract as feed().
        """
        if not self._buffering or not self._pending:
            return "", []

        # Treat the incomplete buffered content as a single raw tool-call block.
        leftover = self._pending
        self._pending = ""
        self._buffering = False
        self._buffer_mode = "xml"
        self._brace_depth = 0
        return "", [leftover]
