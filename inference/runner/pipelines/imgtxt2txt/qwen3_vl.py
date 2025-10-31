"""
Qwen3 VL pipeline for multimodal (text + image) generation.
Optimized for Qwen3 VL models with vision capabilities.
"""

import os
import json
from typing import Dict, Any, Optional, Type, List
from llama_cpp.llama import Llama
from llama_cpp.llama_chat_format import LlamaChatCompletionHandler, Qwen25VLChatHandler
from pydantic import BaseModel  # noqa: F401

# llama_cpp imported lazily within methods to reduce unnecessary top-level dependencies
# Pillow not required for text-only stabilization; multimodal image loading currently disabled.

from models import Model, ModelProfile
from runner.pipelines.llamacpp import BaseLlamaCppPipeline


class Qwen3VLPipeline(BaseLlamaCppPipeline):
    """Qwen3 VL multimodal chat model implementation."""

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        grammar: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ):
        self._multimodal_chat_handler = None
        super().__init__(model, profile, grammar, **kwargs)

    def _initialize_llama(
        self, gguf_path: str, handler: LlamaChatCompletionHandler | None = None
    ) -> Llama:  # type: ignore[override]
        """Attach Qwen25VLChatHandler only when ENABLE_QWEN_VISION_HANDLER=true.

        By default, vision is disabled to avoid segfaults; enabling env var forces
        handler attachment. Set SKIP_QWEN_VISION_HANDLER=true to explicitly skip.
        """
        clip_path = getattr(self.model.details, "clip_model_path", None)
        vision_enabled = os.getenv("ENABLE_QWEN_VISION_HANDLER", "false").lower() == "true"
        vision_skipped = os.getenv("SKIP_QWEN_VISION_HANDLER", "false").lower() == "true"
        if clip_path and vision_enabled and not vision_skipped:
            vh = Qwen25VLChatHandler(clip_model_path=clip_path, verbose=True)
            return super()._initialize_llama(gguf_path, vh)
        return super()._initialize_llama(gguf_path, handler)

    def _get_chat_format(self) -> Optional[str]:  # noqa: D401
        return "chatml"

    # --- Multimodal extensions ---
    def _extract_image_paths(self, messages) -> List[str]:
        paths: List[str] = []
        pattern = "<image:"  # simplistic pattern; expects <image:/abs/or/relative/path>
        for m in messages:
            content = getattr(m, "content", "")
            if not isinstance(content, str):
                continue
            idx = 0
            while True:
                start = content.find(pattern, idx)
                if start == -1:
                    break
                end = content.find(">", start)
                if end == -1:
                    break
                path = content[start + len(pattern) : end].strip()
                if path:
                    paths.append(path)
                idx = end + 1
        return paths

    def _prepare_multimodal_messages(
        self, messages, image_paths: List[str]
    ) -> List[Dict[str, str]]:
        """Embed image path markers into the first system message for the Qwen VL chat handler.

        The upstream handler expects images referenced in the prompt. We inline markers:
        <img src="/path/to/image" />
        This avoids passing unsupported images kwarg.
        """
        llama_messages = self._format_messages_for_llama(messages)
        if image_paths:
            tag_block = "\n".join([f'<img src="{p}" />' for p in image_paths if p])
            # Prepend to first system message if exists, else create one
            if llama_messages and llama_messages[0]["role"] == "system":
                llama_messages[0]["content"] = (
                    tag_block + "\n" + llama_messages[0]["content"]
                )
            else:
                llama_messages.insert(0, {"role": "system", "content": tag_block})
        return llama_messages

    def _get_res(
        self,
        messages,
        stop: Optional[List[str]] = None,
        tools: Optional[List[Any]] = None,
        stream: bool = False,
    ):
        """Extend base response retrieval to include images for multimodal."""
        image_paths = self._extract_image_paths(messages)
        converted_tools = self._convert_tools_to_simple_format(tools)
        llama_messages = self._prepare_multimodal_messages(messages, image_paths)
        self._logger.info(
            f"Chat completion (VL): model={self.model.name}, messages={len(llama_messages)}, tools={len(converted_tools) if converted_tools else 0}, image_paths={len(image_paths)}"
        )
        response_format = None
        grammar = None
        if self.grammar:
            response_format = {
                "type": "json_object",
                "schema": self.grammar.model_json_schema(),
            }
            try:
                from llama_cpp import llama_grammar as _llama_grammar

                grammar = _llama_grammar.LlamaGrammar.from_json_schema(
                    json.dumps(self.grammar.model_json_schema())
                )
            except Exception as e:
                self._logger.warning(f"Failed to build grammar for VL pipeline: {e}")
                grammar = None
        kwargs = {
            "messages": llama_messages,  # type: ignore
            "temperature": self.profile.parameters.temperature or 0.7,
            "top_p": self.profile.parameters.top_p or 0.95,
            "top_k": self.profile.parameters.top_k or 40,
            "stream": stream,
            "stop": self.profile.parameters.stop or stop,
            "max_tokens": self.profile.parameters.max_tokens or 4096,
            "repeat_penalty": self.profile.parameters.repeat_penalty or 1.05,
        }
        if converted_tools:
            kwargs["tools"] = converted_tools  # type: ignore
            kwargs["tool_choice"] = "auto"
        if response_format:
            kwargs["response_format"] = response_format  # type: ignore
        if grammar:
            kwargs["grammar"] = grammar
        # No direct images kwarg; image references embedded in messages
        if not getattr(self, "llama_instance", None):
            raise RuntimeError("Llama instance not initialized for Qwen3VLPipeline")
        return self.llama_instance.create_chat_completion(**kwargs)  # type: ignore


__all__ = ["Qwen3VLPipeline"]
