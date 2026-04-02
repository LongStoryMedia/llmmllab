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
        before = before.lstrip()
        if before.startswith("<think>"):
            before = before[len("<think>") :]
        return before.strip(), after.lstrip("\n"), True
    if not think_closed:
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
        self._pending: str = ""  # accumulated XML waiting for close tag
        self._buffering: bool = False

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
            - complete_blocks: zero or more complete raw XML tool-call strings
        """
        safe_prefix = ""
        if not self._buffering:
            # Not currently inside a raw tool call block.
            open_match = _RAW_TOOL_CALL_RE.search(text)
            if open_match is None:
                # Fast path: no tool-call XML at all, pass through verbatim.
                return text, []

            # An opening tag was found.  Everything before it is safe text;
            # from the tag onwards we start buffering.
            safe_prefix = text[: open_match.start()]
            self._pending = text[open_match.start() :]
            self._buffering = True
            # Fall through to the buffering logic below to handle the case
            # where the close tag is in the same chunk.

        else:
            # Already buffering — append new chunk.
            self._pending += text

        # Try to find a complete block (close tag present).
        complete_blocks: List[str] = []
        safe_text_parts: List[str] = []

        while self._buffering and self._pending:
            close_match = _RAW_TOOL_CALL_CLOSE_RE.search(self._pending)
            if close_match is None:
                # Close tag not yet received — keep buffering, nothing to emit.
                break

            # Complete block found.
            block_end = close_match.end()
            complete_blocks.append(self._pending[:block_end])
            remainder = self._pending[block_end:]
            self._pending = ""
            self._buffering = False

            # The remainder may itself start another tool-call block.
            next_open = _RAW_TOOL_CALL_RE.search(remainder)
            if next_open is None:
                # Remaining text is plain content.
                safe_text_parts.append(remainder)
            else:
                # Plain prefix before next block.
                safe_text_parts.append(remainder[: next_open.start()])
                self._pending = remainder[next_open.start() :]
                self._buffering = True
                # Loop again to see if the next block is also complete.

        # Combine any safe prefix captured before we entered buffering mode
        # (stored in safe_prefix local when we first detected the open tag).
        # NOTE: safe_prefix only exists in the non-buffering entry path above;
        # use getattr on locals isn't idiomatic — instead we rely on the fact
        # that safe_text_parts collects all safe text discovered after the
        # initial open-tag detection.
        final_safe = "".join(safe_text_parts)

        # If we entered this call in non-buffering mode and had a safe prefix,
        # prepend it.  safe_prefix is defined in the branch above; use a flag.
        try:
            final_safe = safe_prefix + final_safe  # type: ignore[name-defined]
        except NameError:
            pass  # We entered while already buffering; no separate safe prefix.

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
        return "", [leftover]
