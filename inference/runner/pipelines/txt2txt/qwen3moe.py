"""
Qwen3 MoE pipeline as BaseChatModel implementation.
Provides custom model-specific optimizations for Qwen MoE models.
"""

from typing import Dict, Any, Optional, Type, List
from pydantic import BaseModel

from models import Model, ModelProfile
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

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        grammar: Optional[Type[BaseModel]] = None,
        **kwargs
    ):
        super().__init__(model, profile, grammar, **kwargs)

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "qwen3-moe-llamacpp"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        base_params = super()._identifying_params
        base_params.update(
            {
                "model_type": "qwen3-moe",
                "chat_format": "chatml",
            }
        )
        return base_params

    def _extract_text_content(self, content) -> str:
        """
        Extract text content from either string or multimodal list format.
        
        LangChain messages can have content as:
        - str: Simple text content
        - List[Dict]: Multimodal content with 'type' and 'text'/'image_url' fields
        
        For text-only models like Qwen3-Coder, we need to extract just the text.
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Extract text from multimodal format
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            return " ".join(text_parts)
        else:
            # Fallback: convert to string
            return str(content)

    def _format_messages_for_llama(self, messages: List) -> List[Dict[str, str]]:
        """Override to ensure proper text extraction for text-only models."""
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
        
        llama_messages = []

        for message in messages:
            # Extract text content properly for each message type
            text_content = self._extract_text_content(message.content)
            
            if isinstance(message, SystemMessage):
                llama_messages.append({"role": "system", "content": text_content})
            elif isinstance(message, HumanMessage):
                llama_messages.append({"role": "user", "content": text_content})
            elif isinstance(message, AIMessage):
                llama_messages.append({"role": "assistant", "content": text_content})
            elif isinstance(message, ToolMessage):
                # Format tool results as user messages
                llama_messages.append(
                    {"role": "user", "content": f"Tool result: {text_content}"}
                )
            else:
                # Fallback: treat as user message
                llama_messages.append({"role": "user", "content": text_content})

        return llama_messages


__all__ = ["Qwen3Moe"]
