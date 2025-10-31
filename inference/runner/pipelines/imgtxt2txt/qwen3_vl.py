"""
Qwen3 VL pipeline for multimodal (text + image) generation.
Optimized for Qwen3 VL models with vision capabilities.
"""

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

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "qwen3-vl-llamacpp"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        base_params = super()._identifying_params
        base_params.update(
            {
                "model_type": "qwen3-vl",
                "vision_capable": True,
                "multimodal": True,
                "chat_format": "chatml",
                "supports_thinking": True,
            }
        )
        return base_params

    def _initialize_llama(
        self, gguf_path: str, handler: LlamaChatCompletionHandler | None = None
    ) -> Llama:
        if self.model.details.clip_model_path:
            handler = Qwen25VLChatHandler(
                clip_model_path=self.model.details.clip_model_path, verbose=True
            )
            return super()._initialize_llama(
                gguf_path,
                handler,
            )
        else:
            return super()._initialize_llama(gguf_path, handler)

    def _get_chat_format(self) -> Optional[str]:  # noqa: D401
        """Return chat format string for llama initialization."""
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

    # def _get_res(
    #     self,
    #     messages,
    #     stop: Optional[List[str]] = None,
    #     tools: Optional[List[Any]] = None,
    #     stream: bool = False,
    # ):
    #     """Return response with vision support while bypassing segfaulting chat path.

    #     llama.cpp segfaults when calling create_chat_completion with Qwen25VLChatHandler.
    #     Root cause isolated: handler + chat_completion path only; plain completion works.

    #     Strategy:
    #     1. Detect if vision handler is attached (chat_handler instance of Qwen25VLChatHandler)
    #     2. Manually build a prompt using the model's chat template semantics (simplified)
    #     3. Inline image markers extracted from messages (<img src="..." />)
    #     4. Support tools & grammar by embedding system instructions and letting llama produce JSON if grammar provided.
    #     5. Use llama_instance.create_completion for generation (stable path).
    #     6. Streaming: emulate streaming by chunking tokens from create_completion(stream=True) when available.
    #     """

    #     if not getattr(self, "llama_instance", None):
    #         raise RuntimeError("Llama instance not initialized for Qwen3VLPipeline")

    #     # If no handler or we explicitly force normal path, fall back to parent behavior
    #     handler = getattr(self.llama_instance, "chat_handler", None)
    #     use_fallback_plain = isinstance(handler, Qwen25VLChatHandler)

    #     image_paths = self._extract_image_paths(messages)
    #     converted_tools = self._convert_tools_to_simple_format(tools)
    #     llama_messages = self._prepare_multimodal_messages(messages, image_paths)

    #     if not use_fallback_plain:
    #         # Normal path (no vision handler): reuse BaseLlamaCppPipeline style create_chat_completion
    #         response_format = None
    #         grammar_obj = None
    #         if self.grammar:
    #             response_format = {
    #                 "type": "json_object",
    #                 "schema": self.grammar.model_json_schema(),
    #             }
    #             try:
    #                 from llama_cpp import llama_grammar as _llama_grammar

    #                 grammar_obj = _llama_grammar.LlamaGrammar.from_json_schema(
    #                     json.dumps(self.grammar.model_json_schema())
    #                 )
    #             except Exception as e:
    #                 self._logger.warning(
    #                     f"Failed to build grammar for VL pipeline: {e}"
    #                 )
    #                 grammar_obj = None
    #         kwargs = {
    #             "messages": llama_messages,  # type: ignore
    #             "temperature": self.profile.parameters.temperature or 0.7,
    #             "top_p": self.profile.parameters.top_p or 0.95,
    #             "top_k": self.profile.parameters.top_k or 40,
    #             "stream": stream,
    #             "stop": self.profile.parameters.stop or stop,
    #             "max_tokens": self.profile.parameters.max_tokens or 4096,
    #             "repeat_penalty": self.profile.parameters.repeat_penalty or 1.05,
    #         }
    #         if converted_tools:
    #             kwargs["tools"] = converted_tools  # type: ignore
    #             kwargs["tool_choice"] = "auto"
    #         if response_format:
    #             kwargs["response_format"] = response_format  # type: ignore
    #         if grammar_obj:
    #             kwargs["grammar"] = grammar_obj
    #         self._logger.info(
    #             f"Chat completion (standard): model={self.model.name}, messages={len(llama_messages)}, tools={len(converted_tools) if converted_tools else 0}, image_paths={len(image_paths)}"
    #         )
    #         return self.llama_instance.create_chat_completion(**kwargs)  # type: ignore

    #     # Fallback: build plain prompt
    #     sys_prompt = ""
    #     user_turns: List[str] = []
    #     for m in llama_messages:
    #         role = m.get("role")
    #         content = m.get("content", "")
    #         if role == "system":
    #             sys_prompt += content + "\n"
    #         elif role == "user":
    #             user_turns.append(content)
    #         elif role == "assistant":
    #             user_turns.append(f"Assistant: {content}")

    #     # Inline tool schema if present
    #     if converted_tools:
    #         tool_block = ["# Tools", "You may call functions in JSON form:"]
    #         for t in converted_tools:
    #             fn = t.get("function", {})
    #             name = fn.get("name", "unknown")
    #             desc = fn.get("description", "")
    #             tool_block.append(f"- {name}: {desc}")
    #         sys_prompt = (sys_prompt + "\n" + "\n".join(tool_block)).strip()

    #     # Image markers already embedded in first system message by _prepare_multimodal_messages
    #     prompt_parts = []
    #     if sys_prompt:
    #         prompt_parts.append(sys_prompt.strip())
    #     for ut in user_turns:
    #         prompt_parts.append(f"User: {ut}")
    #     prompt_parts.append("Assistant:")
    #     final_prompt = "\n".join(prompt_parts)

    #     self._logger.info(
    #         f"Plain completion (vision fallback): model={self.model.name}, prompt_len={len(final_prompt)}, messages={len(llama_messages)}, images={len(image_paths)}"
    #     )

    #     completion_kwargs = {
    #         "prompt": final_prompt,
    #         "temperature": self.profile.parameters.temperature or 0.7,
    #         "top_p": self.profile.parameters.top_p or 0.95,
    #         "top_k": self.profile.parameters.top_k or 40,
    #         "max_tokens": self.profile.parameters.max_tokens or 4096,
    #         "stop": self.profile.parameters.stop or stop,
    #         "repeat_penalty": self.profile.parameters.repeat_penalty or 1.05,
    #         "stream": stream,
    #     }

    #     # Grammar: llama.cpp create_completion expects grammar object; no response_format support here.
    #     if self.grammar:
    #         try:
    #             from llama_cpp import llama_grammar as _llama_grammar

    #             completion_kwargs["grammar"] = (
    #                 _llama_grammar.LlamaGrammar.from_json_schema(
    #                     json.dumps(self.grammar.model_json_schema())
    #                 )
    #             )
    #         except Exception as e:
    #             self._logger.warning(f"Failed to attach grammar on fallback path: {e}")

    #     # Analyzer sometimes flags llama_instance as possibly None; guard explicitly.
    #     llama_ref = getattr(self, "llama_instance", None)
    #     if llama_ref is None:
    #         raise RuntimeError(
    #             "llama_instance missing before completion (vision fallback)"
    #         )
    #     if stream:
    #         return llama_ref.create_completion(**completion_kwargs)
    #     return llama_ref.create_completion(**completion_kwargs)


__all__ = ["Qwen3VLPipeline"]
