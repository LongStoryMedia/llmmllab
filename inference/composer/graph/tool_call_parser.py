"""Tool call parsing utilities for workflow executor."""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel

from models import ToolCall


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


class RawToolCallParser:
    """Parser for raw tool-call XML that models sometimes emit inline."""

    def __init__(self):
        """Initialize the raw tool call parser."""
        pass

    def strip_raw_tool_calls(self, content: str) -> Tuple[str, List[ToolCall]]:
        """
        Strip raw tool-call XML and parse tool calls from it.

        When the model generates text followed by an inline tool call in
        XML format (e.g. ``<tool_call>func_name<arg_key>…``), llama.cpp
        may not recognise the structured tool call and returns everything
        as plain content. Strip everything from the first raw tool-call
        tag onwards, parse tool calls from the stripped portion, and
        return both the cleaned content and extracted tool calls.

        Args:
            content: Content that may contain raw tool-call XML

        Returns:
            Tuple of (cleaned_content, list_of_tool_calls)
        """
        match = _RAW_TOOL_CALL_RE.search(content)
        if not match:
            return content, []

        cleaned = content[: match.start()].rstrip()
        raw_portion = content[match.start() :]
        stripped_len = len(content) - len(cleaned)

        # Parse tool calls from the stripped XML
        parsed_tcs = self._parse_raw_tool_calls(raw_portion)

        return cleaned, parsed_tcs

    def _parse_raw_tool_calls(self, raw: str) -> List[ToolCall]:
        """
        Parse tool calls from raw XML in GLM native or JSON format.

        Handles two formats:
        1. GLM XML: ``<tool_call>func<arg_key>key</arg_key><arg_value>val</arg_value></tool_call>``
        2. JSON:    ``<tool_call>{"name": "func", "arguments": {...}}</tool_call>``

        Args:
            raw: Raw string containing tool call XML

        Returns:
            List of parsed ToolCall objects
        """
        parsed: List[ToolCall] = []

        for block_match in _TOOL_CALL_BLOCK_RE.finditer(raw):
            body = block_match.group(1).strip()
            if not body:
                continue

            # --- Try JSON format first ---
            if body.startswith("{"):
                try:
                    data = json.loads(body)
                    if isinstance(data, dict) and "name" in data:
                        args = data.get("arguments", data.get("args", {}))
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {"raw": args}
                        parsed.append(
                            ToolCall(
                                name=data["name"],
                                args=args if isinstance(args, dict) else {},
                                execution_id=f"raw_{data['name']}_{len(parsed)}",
                                created_at=datetime.now(),
                            )
                        )
                        continue
                except json.JSONDecodeError:
                    pass

            # --- GLM XML format: func_name<arg_key>key</arg_key><arg_value>val</arg_value> ---
            arg_key_pos = body.find("<arg_key>")
            if arg_key_pos == -1:
                continue

            func_name = body[:arg_key_pos].strip()
            if not func_name:
                continue

            args: Dict[str, Any] = {}
            remaining = body[arg_key_pos:]

            while remaining:
                ks = remaining.find("<arg_key>")
                if ks == -1:
                    break
                ke = remaining.find("</arg_key>", ks)
                if ke == -1:
                    break
                key = remaining[ks + len("<arg_key>") : ke].strip()
                vs = remaining.find("<arg_value>", ke)
                if vs == -1:
                    break
                ve = remaining.find("</arg_value>", vs)
                if ve == -1:
                    # Value extends to end of block (unclosed tag)
                    value = remaining[vs + len("<arg_value>") :]
                    args[key] = value
                    break
                value = remaining[vs + len("<arg_value>") : ve]
                args[key] = value
                remaining = remaining[ve + len("</arg_value>") :]

            if func_name:
                parsed.append(
                    ToolCall(
                        name=func_name,
                        args=args,
                        execution_id=f"raw_{func_name}_{len(parsed)}",
                        created_at=datetime.now(),
                    )
                )

        return parsed
