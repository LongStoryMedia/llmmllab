"""
Simplified Qwen3 MoE pipeline - pure LLM interface, no orchestration.
Replaced 1020 lines of LangGraph complexity with direct LLM calls.
"""

import logging
from typing import List, Optional, AsyncIterator, Dict, Any, Tuple

from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage
# (Removed unused langchain imports from simplified runner pipeline)

from models import (
    MessageContent,
    MessageContentType,
    MessageRole,
    Model,
    Message,
    ChatResponse,
    ModelProfile,
)
from runner.pipelines.base import GrammarInput
from runner.pipelines.llamacpp.simple_base import SimpleLlamaCppPipeline


class Qwen3Moe(SimpleLlamaCppPipeline):
    """
    Simplified Qwen3 MoE pipeline - direct LLM calls with <think> tag processing.

    Features:
    - Direct LlamaCpp initialization
    - Clean message formatting with Qwen chat format
    - Hardware optimization for MoE models
    - Simple <think> tag extraction
    """

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
    ):
        super().__init__(model, profile)
        self._logger = logging.getLogger(self.__class__.__name__)

    async def _create_system_prompt(
        self, tools: Optional[List[BaseTool]] = None
    ) -> str:
        """Create system prompt for Qwen with structured tool descriptions.

        We provide two layers of tool guidance:
        1. A human-readable bullet list
        2. A machine-readable JSON array (functions style) the model can echo in a function_call / tool call format.

        This mirrors the richer guidance from the legacy pipeline while keeping this runner-focused implementation lightweight.
        """
        base_prompt = (
            self.profile.system_prompt
            or (
                "You are Qwen, a careful AI assistant. Always decide if a tool is needed. "
                "If a tool is appropriate, emit structured tool calls."
                "Use JSON structures – do not hallucinate parameters not in the schema."
            )
        )

        if not tools:
            return base_prompt

        bullet_lines: List[str] = []
        json_functions: List[Dict[str, Any]] = []
        for tool in tools:
            bullet_lines.append(f"- {tool.name}: {tool.description}")

            # Build minimal JSON schema (attempt to use args_schema if present)
            params: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}
            if hasattr(tool, "args_schema") and tool.args_schema is not None:  # type: ignore[attr-defined]
                try:
                    schema_model = tool.args_schema  # type: ignore[attr-defined]
                    if hasattr(schema_model, "model_json_schema"):
                        params = schema_model.model_json_schema()  # type: ignore
                except Exception:  # pragma: no cover - defensive
                    pass
            else:
                # Generic fallback parameter
                params["properties"] = {
                    "query": {"type": "string", "description": "Primary input"}
                }
                params["required"] = ["query"]

            json_functions.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": params,
                    },
                }
            )

        import json

        return (
            f"{base_prompt}\n\nAvailable tools:\n"
            + "\n".join(bullet_lines)
            + "\n\nTool function specifications (JSON):\n"
            + json.dumps(json_functions, indent=2)
            + "\n\nWhen you decide to call tools, ALWAYS output one of the supported formats:\n"
              "1) Qwen function_call JSON with name and arguments\n"
              "2) <tool_call> JSON blocks\n"
              "3) <function_call>{...}</function_call> blocks\n"
              "4) A single JSON code block containing a tool_calls array."
        )

    async def _format_messages(
        self, messages: List[Message], tools: Optional[List[BaseTool]] = None
    ) -> str:
        """Format messages using Qwen chat format."""
        formatted_parts = []

        # Add system prompt
        system_prompt = await self._create_system_prompt(tools)
        formatted_parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")

        # Add conversation messages
        for msg in messages:
            content_text = ""
            for content in msg.content:
                if content.type == MessageContentType.TEXT and content.text:
                    content_text += content.text

            if msg.role == MessageRole.USER:
                formatted_parts.append(f"<|im_start|>user\n{content_text}<|im_end|>")
            elif msg.role == MessageRole.ASSISTANT:
                formatted_parts.append(
                    f"<|im_start|>assistant\n{content_text}<|im_end|>"
                )

        # Add assistant start for completion
        formatted_parts.append("<|im_start|>assistant\n")

        return "\n".join(formatted_parts)

    # --- Tool Call Parsing (expanded from legacy pipeline patterns) ---
    def _parse_all_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Parse tool calls from model output supporting multiple legacy & modern patterns.

        Patterns (attempted in order; first successful pattern may still allow multiples):
        1. Qwen style: "function_call": {"name": "...", "arguments": "{...}"}
        2. Qwen style (object args): "function_call": {"name": "...", "arguments": {...}}
        3. XML-like <tool_call>{...}</tool_call>
        4. XML-like <function_call>{...}</function_call>
        5. JSON code fence containing {"tool_calls": [ ... ]}
        6. Standalone JSON objects with name + arguments (fallback)

        Returns canonical list: [{"name": str, "arguments": dict}]
        """
        import re, json

        parsed: List[Dict[str, Any]] = []

        def _add(name: Optional[str], args_raw: Any):
            if not name:
                return
            # Normalize args
            if isinstance(args_raw, str):
                # Try to load JSON string
                try:
                    args_obj = json.loads(args_raw)
                except Exception:
                    # Best-effort: return as wrapper
                    args_obj = {"value": args_raw}
            elif isinstance(args_raw, dict):
                args_obj = args_raw
            else:
                args_obj = {"value": args_raw}
            parsed.append({"name": name, "arguments": args_obj})

        try:
            # Pattern 1: function_call with arguments as JSON string
            fc_pattern_str = re.compile(
                r'"function_call"\s*:\s*\{\s*"name"\s*:\s*"([^"]+)",\s*"arguments"\s*:\s*"(.*?)"\s*\}',
                re.DOTALL,
            )
            for name, args_str in fc_pattern_str.findall(content):
                try:
                    # Unescape common JSON escapes inside the string
                    unescaped = args_str.encode("utf-8").decode("unicode_escape")
                    _add(name, unescaped)
                except Exception:  # pragma: no cover
                    continue

            # Pattern 2: function_call with arguments as object
            fc_pattern_obj = re.compile(
                r'"function_call"\s*:\s*\{\s*"name"\s*:\s*"([^"]+)",\s*"arguments"\s*:\s*(\{.*?\})\s*\}',
                re.DOTALL,
            )
            for name, args_obj in fc_pattern_obj.findall(content):
                try:
                    _add(name, json.loads(args_obj))
                except Exception:
                    continue

            # Pattern 3: <tool_call>{...}</tool_call>
            xml_tool_pattern = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE)
            for block in xml_tool_pattern.findall(content):
                try:
                    data = json.loads(block)
                    _add(data.get("name") or data.get("tool"), data.get("arguments") or data.get("args") or {})
                except Exception:
                    continue

            # Pattern 4: <function_call>{...}</function_call>
            xml_fc_pattern = re.compile(
                r"<function_call>\s*(\{.*?\})\s*</(?:function_call|FunctionCall)>",
                re.DOTALL | re.IGNORECASE,
            )
            for block in xml_fc_pattern.findall(content):
                try:
                    data = json.loads(block)
                    _add(data.get("name"), data.get("arguments") or data.get("args") or {})
                except Exception:
                    continue

            # Pattern 5: Code fence with tool_calls array
            code_fence_pattern = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
            for block in code_fence_pattern.findall(content):
                try:
                    data = json.loads(block)
                    if isinstance(data, dict) and "tool_calls" in data and isinstance(data["tool_calls"], list):
                        for tc in data["tool_calls"]:
                            if isinstance(tc, dict):
                                _add(tc.get("name"), tc.get("arguments") or tc.get("args") or {})
                except Exception:
                    continue

            # Pattern 6: Fallback – attempt to find top-level JSON object with name & arguments
            if not parsed:
                generic_obj_pattern = re.compile(r"(\{\s*\"name\"\s*:\s*\"[^\"]+\".*?\})", re.DOTALL)
                for block in generic_obj_pattern.findall(content):
                    try:
                        data = json.loads(block)
                        if "name" in data and ("arguments" in data or "args" in data):
                            _add(data.get("name"), data.get("arguments") or data.get("args") or {})
                    except Exception:
                        continue
        except Exception as e:  # pragma: no cover - defensive outer catch
            self._logger.error(f"Tool call parsing unexpected failure: {e}")

        # Deduplicate by (name, arguments json-dump) pair preserving order
        seen = set()
        deduped: List[Dict[str, Any]] = []
        from json import dumps as _dumps
        for tc in parsed:
            key = (tc.get("name"), _dumps(tc.get("arguments"), sort_keys=True))
            if key not in seen:
                seen.add(key)
                deduped.append(tc)

        if deduped:
            self._logger.info(f"Parsed {len(deduped)} tool call(s): {[c['name'] for c in deduped]}")
        else:
            self._logger.debug("No tool calls parsed from content.")
        return deduped

    def _clean_tool_call_markup(self, content: str) -> str:
        """Remove known tool call markup patterns from assistant-facing text."""
        import re
        patterns = [
            # function_call JSON (string args)
            r'"function_call"\s*:\s*\{\s*"name"\s*:\s*"[^"\n]+",\s*"arguments"\s*:\s*".*?"\s*\}',
            # function_call JSON (object args)
            r'"function_call"\s*:\s*\{\s*"name"\s*:\s*"[^"\n]+",\s*"arguments"\s*:\s*\{.*?\}\s*\}',
            # <tool_call> blocks
            r'<tool_call>\s*\{.*?\}\s*</tool_call>',
            # <function_call> blocks
            r'<function_call>\s*\{.*?\}\s*</(?:function_call|FunctionCall)>',
            # code fence with tool_calls
            r'```json\s*\{.*?"tool_calls".*?\}\s*```',
        ]
        cleaned = content
        for p in patterns:
            cleaned = re.sub(p, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        # Collapse excessive blank lines
        cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)
        return cleaned.strip()

    def _extract_response_content(self, raw_response: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract visible assistant content & structured tool calls.

        Order of operations:
        1. Parse tool calls (preserve JSON before modifications)
        2. Remove think tags
        3. Strip tool call markup from visible content
        4. Normalize whitespace
        """
        import re

        tool_calls = self._parse_all_tool_calls(raw_response)

        # Remove think tags (retain reasoning optionally later if needed)
        no_think = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL)

        # Remove tool call markup for user-facing text
        cleaned = self._clean_tool_call_markup(no_think)
        cleaned = re.sub(r"\n\s*\n", "\n", cleaned).strip()

        # Fallback to raw if cleaning produced empty but there are no tool calls
        if not cleaned and not tool_calls:
            cleaned = raw_response.strip()
        return cleaned, tool_calls

    async def invoke(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[GrammarInput] = None,
        **kwargs,
    ) -> ChatResponse:
        """Invoke the Qwen LLM directly."""
        _ = grammar, kwargs  # Suppress unused warnings

        # Initialize LLM if needed
        if self.llm is None:
            self.llm = self._initialize_llm()

        try:
            # Format conversation
            formatted_prompt = await self._format_messages(messages, tools)

            # Invoke LLM directly
            if self.llm is None:
                raise RuntimeError("LLM not initialized")
            response = await self.llm.ainvoke([HumanMessage(content=formatted_prompt)])

            # Extract and clean content
            raw_content = str(response.content) if response.content else ""
            cleaned_content, tool_calls = self._extract_response_content(raw_content)

            # Create response message
            result_message = Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(type=MessageContentType.TEXT, text=cleaned_content)
                ],
                tool_calls=tool_calls if tool_calls else None,
            )

            return ChatResponse(done=True, message=result_message)

        except Exception as e:
            self._logger.error(f"Qwen LLM invocation failed: {e}")
            error_msg = f"Error: {str(e)}"
            error_message = Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(type=MessageContentType.TEXT, text=error_msg)],
            )
            return ChatResponse(done=True, message=error_message)

    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[GrammarInput] = None,
        **kwargs,
    ) -> AsyncIterator[ChatResponse]:
        """Stream responses from Qwen LLM."""
        _ = grammar, kwargs  # Suppress unused warnings

        # Initialize LLM if needed
        if self.llm is None:
            self.llm = self._initialize_llm()

        try:
            # Format conversation
            formatted_prompt = await self._format_messages(messages, tools)

            # Stream from LLM
            if self.llm is None:
                raise RuntimeError("LLM not initialized")

            accumulated_content = ""
            async for chunk in self.llm.astream([HumanMessage(content=formatted_prompt)]):
                if hasattr(chunk, "content") and chunk.content:
                    chunk_text = str(chunk.content)
                    accumulated_content += chunk_text

                    # For streaming, we send raw chunks and clean at the end
                    chunk_message = Message(
                        role=MessageRole.ASSISTANT,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT, text=chunk_text
                            )
                        ],
                    )
                    yield ChatResponse(done=False, message=chunk_message)

            # Final chunk to indicate completion
            # On completion parse accumulated content for tool calls
            cleaned_content, tool_calls = self._extract_response_content(accumulated_content)
            final_message = Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(type=MessageContentType.TEXT, text=cleaned_content)],
                tool_calls=tool_calls if tool_calls else None,
            )
            yield ChatResponse(done=True, message=final_message)

        except Exception as e:
            self._logger.error(f"Qwen LLM streaming failed: {e}")
            error_msg = f"Error: {str(e)}"
            error_message = Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(type=MessageContentType.TEXT, text=error_msg)],
            )
            yield ChatResponse(done=True, message=error_message)


# ---------------------------------------------------------------------------
# Backward compatibility alias expected by pipeline_factory & simple_factory
# The factories import QwenSimplePipeline; original refactor renamed the class
# to Qwen3Moe causing ImportError and pipeline creation failure. We provide a
# thin alias to restore compatibility without altering external references.
# ---------------------------------------------------------------------------
class QwenSimplePipeline(Qwen3Moe):  # type: ignore
    """Backward compatible alias for Qwen3 MoE text generation pipeline."""
    # (No additional implementation needed)

__all__ = [
    "Qwen3Moe",
    "QwenSimplePipeline",
]
