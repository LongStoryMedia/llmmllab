"""
Simplified Qwen3 MoE pipeline - pure LLM interface, no orchestration.
Replaced 1020 lines of LangGraph complexity with direct LLM calls.
"""

import os
import logging
import asyncio
from typing import List, Optional, AsyncIterator, Dict, Any, Tuple

from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_community.chat_models.llamacpp import ChatLlamaCpp
from langchain_core.callbacks import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

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
        """Create system prompt for Qwen with tool descriptions."""
        base_prompt = (
            self.profile.system_prompt or "You are Qwen, a helpful AI assistant."
        )

        if not tools:
            return base_prompt

        # Create tool descriptions for Qwen format
        tool_descriptions = []
        for tool in tools:
            tool_desc = f"- {tool.name}: {tool.description}"
            tool_descriptions.append(tool_desc)

        tools_section = "Available tools:\n" + "\n".join(tool_descriptions)

        return f"{base_prompt}\n\n{tools_section}"

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

    def _parse_tool_calls(self, content: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract <tool_call> JSON blocks and return remaining content + parsed calls.

        Expected pattern:
        <tool_call>\n{"name": "web_search", "arguments": {...}}\n</tool_call>
        Multiple blocks allowed.
        """
        import re, json

        tool_calls: List[Dict[str, Any]] = []

        def repl(match: re.Match) -> str:
            block = match.group(1).strip()
            try:
                # Some models may wrap code fences; strip them
                block_stripped = block.strip('`')
                parsed = json.loads(block_stripped)
                # Normalize keys
                name = parsed.get("name") or parsed.get("tool")
                args = parsed.get("arguments") or parsed.get("args") or {}
                if name:
                    tool_calls.append({"name": name, "arguments": args})
            except Exception:
                pass  # Silently ignore malformed tool JSON in this pass
            return ""  # Remove the block from assistant visible content

        # Capture blocks between tags (non-greedy)
        pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
        remaining = re.sub(pattern, repl, content)
        return remaining, tool_calls

    def _extract_response_content(self, raw_response: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract response content, parse tool calls, and handle <think> tags."""
        import re

        # First parse tool calls so think tag removal doesn't disturb JSON
        without_tools, tool_calls = self._parse_tool_calls(raw_response)

        # Remove think tags and their content
        cleaned = re.sub(r"<think>.*?</think>", "", without_tools, flags=re.DOTALL)
        cleaned = re.sub(r"\n\s*\n", "\n", cleaned).strip()
        return (cleaned or raw_response, tool_calls)

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
    pass

__all__ = [
    "Qwen3Moe",
    "QwenSimplePipeline",
]
