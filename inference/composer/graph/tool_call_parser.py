"""Tool call parsing utilities for workflow executor."""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

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

# Qwen / hermes-style: <function=FuncName> ... </function>
# The function name is in the tag attribute, parameters follow as
# <parameter=key>value</parameter> pairs.
_FUNCTION_TAG_RE = re.compile(
    r"<function=([^>]+)>",
    re.IGNORECASE,
)
_PARAMETER_RE = re.compile(
    r"<parameter=([^>]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)",
    re.IGNORECASE | re.DOTALL,
)


class RawToolCallParser:
    """Parser for raw tool-call XML that models sometimes emit inline.

    Handles three formats that appear in the wild:

    1. **JSON** — ``<tool_call>{"name": "func", "arguments": {...}}</tool_call>``
    2. **GLM XML** — ``<tool_call>func_name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>``
    3. **Qwen / hermes** — ``<tool_call><function=FuncName><parameter=key>val</parameter></tool_call>``
    """

    def strip_raw_tool_calls(self, content: str) -> Tuple[str, List[ToolCall]]:
        """
        Strip raw tool-call XML and parse tool calls from it.

        When the model generates text followed by an inline tool call in
        XML format, llama.cpp may not recognise the structured tool call
        and returns everything as plain content.  Strip everything from
        the first raw tool-call tag onwards, parse tool calls from the
        stripped portion, and return both the cleaned content and
        extracted tool calls.

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

        parsed_tcs = self._parse_raw_tool_calls(raw_portion)
        return cleaned, parsed_tcs

    def _parse_raw_tool_calls(self, raw: str) -> List[ToolCall]:
        """
        Parse tool calls from raw XML in GLM native, JSON, or Qwen format.

        Handles three formats:

        1. JSON:
           ``<tool_call>{"name": "func", "arguments": {...}}</tool_call>``

        2. GLM XML:
           ``<tool_call>func<arg_key>key</arg_key><arg_value>val</arg_value></tool_call>``

        3. Qwen / hermes ``<function=>`` style:
           ``<tool_call><function=FuncName><parameter=key>val</parameter></tool_call>``

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

            tc = (
                self._try_parse_json(body, len(parsed))
                or self._try_parse_function_tag(body, len(parsed))
                or self._try_parse_glm_xml(body, len(parsed))
            )
            if tc is not None:
                parsed.append(tc)

        return parsed

    # ------------------------------------------------------------------
    # Format-specific parsers
    # ------------------------------------------------------------------

    def _try_parse_json(self, body: str, index: int) -> ToolCall | None:
        """Parse JSON-format tool call body.  Returns None if not applicable."""
        if not body.startswith("{"):
            return None
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict) or "name" not in data:
            return None

        args = data.get("arguments", data.get("args", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}

        return ToolCall(
            name=data["name"],
            args=args if isinstance(args, dict) else {},
            execution_id=f"raw_{data['name']}_{index}",
            created_at=datetime.now(),
        )

    def _try_parse_function_tag(self, body: str, index: int) -> ToolCall | None:
        """Parse Qwen/hermes ``<function=Name><parameter=key>val</parameter>`` format.

        Returns None if not applicable.
        """
        func_match = _FUNCTION_TAG_RE.match(body)
        if not func_match:
            return None

        func_name = func_match.group(1).strip()
        if not func_name:
            return None

        args: Dict[str, Any] = {}
        for param_match in _PARAMETER_RE.finditer(body):
            key = param_match.group(1).strip()
            value = param_match.group(2).strip()
            if key:
                args[key] = value

        return ToolCall(
            name=func_name,
            args=args,
            execution_id=f"raw_{func_name}_{index}",
            created_at=datetime.now(),
        )

    def _try_parse_glm_xml(self, body: str, index: int) -> ToolCall | None:
        """Parse GLM XML ``func_name<arg_key>k</arg_key><arg_value>v</arg_value>`` format.

        Returns None if not applicable.
        """
        arg_key_pos = body.find("<arg_key>")
        if arg_key_pos == -1:
            return None

        func_name = body[:arg_key_pos].strip()
        if not func_name:
            return None

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

        return ToolCall(
            name=func_name,
            args=args,
            execution_id=f"raw_{func_name}_{index}",
            created_at=datetime.now(),
        )
