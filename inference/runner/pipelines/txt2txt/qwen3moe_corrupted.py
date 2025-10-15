"""
Qwen3 MoE pipeline as BaseChatModel implementation.
Provides custom model-specific optimizations for Qwen MoE models.
"""

import logging
import re
from typing import List, Optional

from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage

from models import Model, ModelProfile
from runner.pipelines.base import GrammarInput
from runner.pipelines.llamacpp import BaseLlamaCppPipeline


class Qwen3Moe(BaseLlamaCppPipeline):
    """
    Qwen3 MoE chat model implementation.

    Features:
    - Optimized for Qwen3 MoE models (e.g., Qwen2.5-Coder-32B-Instruct)
    - Custom chat format for Qwen models
    - Hardware optimization for MoE architecture
    - <think> tag processing for reasoning models
    """

    def __init__(self, model: Model, profile: ModelProfile, **kwargs):
        super().__init__(model, profile, **kwargs)
        self._logger = logging.getLogger(self.__class__.__name__)

import os
import logging
import asyncio
from typing import List, Optional, AsyncIterator

from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from langchain.agents import create_agent

# (Removed unused langchain imports from simplified runner pipeline)

from models import (
    LangChainMessage,
    MessageContent,
    MessageContentType,
    MessageRole,
    Model,
    Message,
    ChatResponse,
    ModelProfile,
)
from runner.pipelines.base import GrammarInput
from runner.pipelines.llamacpp import BaseLlamaCppPipeline


class Qwen3Moe(BaseLlamaCppPipeline):
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

    def _extract_response_content(self, raw_response: str) -> str:
        """Extract response content and handle <think> tags."""
        # Remove <think>...</think> blocks for cleaner output
        import re

        # Remove think tags and their content
        cleaned = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL)

        # Clean up extra whitespace
        cleaned = re.sub(r"\n\s*\n", "\n", cleaned)
        cleaned = cleaned.strip()

        return cleaned or raw_response  # Fallback to original if nothing left

    def _parse_tool_calls(self, content: str) -> List[dict]:
        """Parse tool calls from XML format."""
        import json
        import re

        tool_calls = []

        # Look for <tool_call> XML tags - handle multiline JSON
        tool_call_pattern = r"<tool_call>\s*(\{[^<]*?\})\s*</tool_call>"
        matches = re.findall(tool_call_pattern, content, re.DOTALL | re.IGNORECASE)

        self._logger.debug(f"Parsing tool calls from content: {content[:500]}...")
        self._logger.debug(f"Found {len(matches)} potential tool call matches")

        for i, match in enumerate(matches):
            try:
                # Parse the JSON content
                tool_data = json.loads(match)

                if "name" in tool_data:
                    formatted_call = {
                        "name": tool_data["name"],
                        "args": tool_data.get("arguments", {}),
                        "id": f"call_{i}_{tool_data['name']}",
                        "type": "tool_call",
                    }
                    tool_calls.append(formatted_call)
                    self._logger.debug(f"Parsed XML tool call: {formatted_call}")
                else:
                    self._logger.warning(
                        f"Tool call missing 'name' field: {match[:100]}..."
                    )

            except (json.JSONDecodeError, KeyError) as e:
                self._logger.warning(
                    f"Failed to parse XML tool call from: {match[:100]}... Error: {e}"
                )
                continue

        self._logger.debug(f"Returning {len(tool_calls)} parsed tool calls")
        return tool_calls

    async def invoke(
        self,
        messages: List[LangChainMessage],
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[GrammarInput] = None,
        metadata: Optional[dict] = None,
        **kwargs,
    ) -> ChatResponse:
        """Invoke the Qwen LLM directly."""
        _ = grammar, kwargs  # Suppress unused warnings

        # Initialize LLM if needed
        if self.llm is None:
            self.llm = self._initialize_llm()

        try:
            # Invoke LLM directly
            if self.llm is None:
                raise RuntimeError("LLM not initialized")
            response = await self.llm.ainvoke(
                input=messages,
                config=RunnableConfig(metadata=metadata) if metadata else None,
            )

            # Extract and clean content
            raw_content = str(response.content) if response.content else ""
            cleaned_content = self._extract_response_content(raw_content)

            # Parse tool calls from raw content
            self._logger.info(
                f"QWEN3MOE INVOKE: tools param = {len(tools) if tools else 'None'}"
            )
            self._logger.info(
                f"QWEN3MOE INVOKE: raw_content preview = {raw_content[:200]}..."
            )
            tool_calls = self._parse_tool_calls(raw_content) if tools else None
            self._logger.info(
                f"QWEN3MOE INVOKE: parsed tool_calls = {len(tool_calls) if tool_calls else 'None'}"
            )

            # Create response message
            result_message = Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(type=MessageContentType.TEXT, text=cleaned_content)
                ],
                tool_calls=tool_calls,
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
        messages: List[LangChainMessage],
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[GrammarInput] = None,
        metadata: Optional[dict] = None,
        **kwargs,
    ) -> AsyncIterator[ChatResponse]:
        """Stream responses from Qwen LLM."""
        _ = grammar, kwargs  # Suppress unused warnings

        self._logger.info(
            f"QWEN3MOE STREAM START: tools param = {len(tools) if tools else 'None'}"
        )

        # Initialize LLM if needed
        if self.llm is None:
            self.llm = self._initialize_llm()

        try:

            # Stream from LLM
            if self.llm is None:
                raise RuntimeError("LLM not initialized")

            accumulated_content = ""
            async for chunk in self.llm.astream(
                input=messages,
                config=RunnableConfig(metadata=metadata) if metadata else None,
            ):
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

            # Final chunk to indicate completion with tool calls
            cleaned_content = self._extract_response_content(accumulated_content)
            self._logger.info(
                f"QWEN3MOE STREAM: tools param = {len(tools) if tools else 'None'}"
            )
            self._logger.info(
                f"QWEN3MOE STREAM: accumulated_content preview = {accumulated_content[:200]}..."
            )
            tool_calls = self._parse_tool_calls(accumulated_content) if tools else None
            self._logger.info(
                f"QWEN3MOE STREAM: parsed tool_calls = {len(tool_calls) if tool_calls else 'None'}"
            )

            final_message = Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(type=MessageContentType.TEXT, text=cleaned_content)
                ],
                tool_calls=tool_calls,
            )

            # Debug: Log final message tool calls
            self._logger.info(
                f"QWEN3MOE STREAM: Final message created with tool_calls={len(final_message.tool_calls) if final_message.tool_calls else 0}"
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
