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

# Bare JSON tool call — model outputs {"name": "...", "arguments": {...}}
# directly in content without any XML wrapper.  Must have at least "name"
# and either "arguments" or "args" to avoid false positives.
_BARE_JSON_TOOL_CALL_RE = re.compile(
    r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"(?:arguments|args)"\s*:',
    re.DOTALL,
)

# Markdown code-block wrapped tool call — ```json\n{...}\n```
_CODE_BLOCK_TOOL_CALL_RE = re.compile(
    r"```(?:json)?\s*\n(\{.*?\})\s*\n```",
    re.DOTALL,
)

# Mistral-style [TOOL_CALLS] prefix
_MISTRAL_TOOL_CALLS_RE = re.compile(
    r"\[TOOL_CALLS\]\s*(\[.*\])",
    re.DOTALL,
)


class RawToolCallParser:
    """Parser for raw tool-call text that models sometimes emit inline.

    Handles multiple formats that appear in the wild:

    1. **XML-wrapped JSON** — ``<tool_call>{"name": "func", "arguments": {...}}</tool_call>``
    2. **GLM XML** — ``<tool_call>func_name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>``
    3. **Qwen / hermes** — ``<tool_call><function=FuncName><parameter=key>val</parameter></tool_call>``
    4. **Bare JSON** — ``{"name": "func", "arguments": {...}}`` (no XML wrapper)
    5. **Code-block JSON** — `` ```json\\n{"name": ...}\\n``` ``
    6. **Mistral** — ``[TOOL_CALLS] [{"name": "func", ...}]``
    """

    def strip_raw_tool_calls(self, content: str) -> Tuple[str, List[ToolCall]]:
        """
        Strip raw tool-call markup and parse tool calls from content.

        Tries XML-tagged formats first, then falls back to bare JSON,
        code-block, and Mistral-style detection.

        Args:
            content: Content that may contain raw tool-call markup

        Returns:
            Tuple of (cleaned_content, list_of_tool_calls)
        """
        # 1. Try XML-tagged formats first (most specific).
        match = _RAW_TOOL_CALL_RE.search(content)
        if match:
            cleaned = content[: match.start()].rstrip()
            raw_portion = content[match.start() :]
            parsed_tcs = self._parse_raw_tool_calls(raw_portion)
            if parsed_tcs:
                return cleaned, parsed_tcs

        # 2. Try Mistral-style [TOOL_CALLS] prefix.
        mistral_match = _MISTRAL_TOOL_CALLS_RE.search(content)
        if mistral_match:
            cleaned = content[: mistral_match.start()].rstrip()
            parsed = self._try_parse_mistral(mistral_match.group(1))
            if parsed:
                return cleaned, parsed

        # 3. Try code-block wrapped JSON.
        code_blocks = list(_CODE_BLOCK_TOOL_CALL_RE.finditer(content))
        if code_blocks:
            parsed = []
            for cb_match in code_blocks:
                tc = self._try_parse_json(cb_match.group(1).strip(), len(parsed))
                if tc:
                    parsed.append(tc)
            if parsed:
                # Strip the code blocks from content.
                cleaned = content
                for cb_match in reversed(code_blocks):
                    cleaned = cleaned[: cb_match.start()] + cleaned[cb_match.end() :]
                return cleaned.rstrip(), parsed

        # 4. Try bare JSON tool calls (least specific — check last to avoid
        #    false positives on normal JSON in conversation).
        bare_match = _BARE_JSON_TOOL_CALL_RE.search(content)
        if bare_match:
            parsed = self._try_parse_bare_json(content, bare_match.start())
            if parsed:
                cleaned = content[: bare_match.start()].rstrip()
                return cleaned, parsed

        return content, []

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

    def _try_parse_bare_json(self, content: str, start: int) -> List[ToolCall]:
        """Parse one or more bare JSON tool calls starting at *start*.

        The model may emit multiple ``{"name": ..., "arguments": ...}``
        objects separated by whitespace or newlines.
        """
        parsed: List[ToolCall] = []
        remaining = content[start:]
        while remaining:
            remaining = remaining.lstrip()
            if not remaining.startswith("{"):
                break
            # Find matching closing brace via simple brace counting.
            depth = 0
            end = -1
            for i, ch in enumerate(remaining):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end == -1:
                break
            tc = self._try_parse_json(remaining[:end].strip(), len(parsed))
            if tc:
                parsed.append(tc)
            remaining = remaining[end:]
        return parsed

    def _try_parse_mistral(self, json_array_str: str) -> List[ToolCall]:
        """Parse Mistral-style ``[TOOL_CALLS] [{...}, ...]`` array."""
        try:
            data = json.loads(json_array_str)
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        parsed: List[ToolCall] = []
        for item in data:
            if isinstance(item, dict) and "name" in item:
                tc = self._try_parse_json(json.dumps(item), len(parsed))
                if tc:
                    parsed.append(tc)
        return parsed
