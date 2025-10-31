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
        """Extract image paths from either structured content or <image:...> tags."""
        paths: List[str] = []
        for m in messages:
            content = getattr(m, "content", None)
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url")
                        if url:
                            paths.append(url)
            elif isinstance(content, str):
                pattern = "<image:"
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

    def _prepare_multimodal_messages(self, messages) -> List[Dict[str, Any]]:
        """Convert LangChain messages to llama.cpp structured chat messages.

        For vision: retain list-of-dict content items (image_url + text). If a message
        contains <image:path> markers in a raw string, convert them to structured parts.
        """
        prepared: List[Dict[str, Any]] = []
        for m in messages:
            role = "user"
            if hasattr(m, "type"):
                if m.type == "human":
                    role = "user"
                elif m.type == "ai":
                    role = "assistant"
                elif m.type == "system":
                    role = "system"
            content = getattr(m, "content", "")
            if isinstance(content, list):
                # Assume already structured
                prepared.append({"role": role, "content": content})
                continue
            if isinstance(content, str):
                # Extract <image:...> markers and build structured list
                parts: List[Dict[str, Any]] = []
                pattern = "<image:"
                idx = 0
                last_end = 0
                while True:
                    start = content.find(pattern, idx)
                    if start == -1:
                        break
                    end = content.find(">", start)
                    if end == -1:
                        break
                    # Text before image
                    if start > last_end:
                        before = content[last_end:start].strip()
                        if before:
                            parts.append({"type": "text", "text": before})
                    path = content[start + len(pattern): end].strip()
                    if path:
                        # Normalize file path url
                        if not path.startswith("file://"):
                            if path.startswith("/"):
                                url = f"file://{path}"
                            else:
                                url = f"file:///{path}"
                        else:
                            url = path
                        parts.append({"type": "image_url", "image_url": {"url": url}})
                    idx = end + 1
                    last_end = idx
                # Remainder text
                if last_end < len(content):
                    tail = content[last_end:].strip()
                    if tail:
                        parts.append({"type": "text", "text": tail})
                if parts:
                    prepared.append({"role": role, "content": parts})
                else:
                    prepared.append({"role": role, "content": [{"type": "text", "text": content}]})
            else:
                prepared.append({"role": role, "content": [{"type": "text", "text": str(content)}]})
        return prepared

    def _get_res(
        self,
        messages,
        stop: Optional[List[str]] = None,
        tools: Optional[List[Any]] = None,
        stream: bool = False,
    ):
        """Vision-aware response path with guarded fallback.

        Primary path: create_chat_completion (vision handler). Fallback: create_completion.
        """
        if not getattr(self, "llama_instance", None):
            raise RuntimeError("Llama instance not initialized for Qwen3VLPipeline")

        handler = getattr(self.llama_instance, "chat_handler", None)
        vision_handler_active = isinstance(handler, Qwen25VLChatHandler)

        converted_tools = self._convert_tools_to_simple_format(tools)
        llama_messages = self._prepare_multimodal_messages(messages)
        image_paths = self._extract_image_paths(messages)

        force_plain = os.getenv("QWEN3_VL_FORCE_PLAIN", "0").lower() == "1"
        force_chat = os.getenv("QWEN3_VL_FORCE_CHAT", "0").lower() == "1"
        disable_grammar = os.getenv("QWEN3_VL_DISABLE_GRAMMAR", "0").lower() == "1"
        disable_tools = os.getenv("QWEN3_VL_DISABLE_TOOLS", "0").lower() == "1"

        def _images_present() -> bool:
            for m in messages:
                content = getattr(m, "content", None)
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "image_url":
                            return True
                elif isinstance(content, str) and "<image:" in content:
                    return True
            return False

        want_chat_path = vision_handler_active and not force_plain and (
            force_chat or _images_present() or True
        )

        def _build_chat_kwargs(msgs: List[Dict[str, Any]]):
            response_format = None
            grammar_obj = None
            if self.grammar and not disable_grammar:
                response_format = {
                    "type": "json_object",
                    "schema": self.grammar.model_json_schema(),
                }
                try:  # pragma: no cover
                    from llama_cpp import llama_grammar as _llama_grammar  # type: ignore
                    grammar_obj = _llama_grammar.LlamaGrammar.from_json_schema(
                        json.dumps(self.grammar.model_json_schema())
                    )
                except Exception as e:  # pragma: no cover
                    self._logger.warning(f"Failed to build grammar: {e}")
                    grammar_obj = None
            kwargs: Dict[str, Any] = {
                "messages": msgs,  # type: ignore
                "temperature": self.profile.parameters.temperature or 0.7,
                "top_p": self.profile.parameters.top_p or 0.95,
                "top_k": self.profile.parameters.top_k or 40,
                "stream": stream,
                "stop": self.profile.parameters.stop or stop,
                "max_tokens": self.profile.parameters.max_tokens or 4096,
                "repeat_penalty": self.profile.parameters.repeat_penalty or 1.05,
            }
            if converted_tools and not disable_tools:
                kwargs["tools"] = converted_tools  # type: ignore
                kwargs["tool_choice"] = "auto"
            if response_format:
                kwargs["response_format"] = response_format  # type: ignore
            if grammar_obj:
                kwargs["grammar"] = grammar_obj
            return kwargs

        if want_chat_path:
            self._logger.info(
                f"Vision chat path: handler_active={vision_handler_active}, images={len(image_paths)}, force_chat={force_chat}, disable_tools={disable_tools}, disable_grammar={disable_grammar}"
            )
            try:
                chat_kwargs = _build_chat_kwargs(llama_messages)
                return self.llama_instance.create_chat_completion(**chat_kwargs)  # type: ignore
            except Exception as vision_error:
                self._logger.error(
                    f"Vision chat path failed, falling back to plain completion: {vision_error}"
                )

        # Fallback plain completion path
        def _flatten(content: Any) -> str:
            if isinstance(content, list):
                out: List[str] = []
                for part in content:
                    if isinstance(part, dict):
                        t = part.get("type")
                        if t == "text":
                            txt = part.get("text", "")
                            if txt:
                                out.append(txt)
                        elif t == "image_url":
                            url = part.get("image_url", {}).get("url", "")
                            if url:
                                out.append(f"[image:{url}]")
                    else:
                        out.append(str(part))
                return " ".join([p for p in out if p])
            return str(content)

        sys_prompt = ""
        dialogue: List[str] = []
        for entry in llama_messages:
            role = entry.get("role")
            content = entry.get("content", "")
            flat = _flatten(content)
            if role == "system" and flat:
                sys_prompt += flat + "\n"
            elif role == "user":
                dialogue.append(f"User: {flat}")
            elif role == "assistant":
                dialogue.append(f"Assistant: {flat}")

        if converted_tools and not disable_tools:
            tool_lines = ["# Tools", "You may call functions by emitting JSON:"]
            for t in converted_tools:
                fn = t.get("function", {})
                name = fn.get("name", "unknown")
                desc = fn.get("description", "")
                tool_lines.append(f"- {name}: {desc}")
            sys_prompt = (sys_prompt + "\n" + "\n".join(tool_lines)).strip()

        prompt_parts: List[str] = []
        if sys_prompt.strip():
            prompt_parts.append(sys_prompt.strip())
        prompt_parts.extend(dialogue)
        prompt_parts.append("Assistant:")
        final_prompt = "\n".join(prompt_parts)

        self._logger.info(
            f"Plain completion (vision fallback): model={self.model.name}, prompt_len={len(final_prompt)}, messages={len(llama_messages)}, images={len(image_paths)}"
        )

        completion_kwargs: Dict[str, Any] = {
            "prompt": final_prompt,
            "temperature": self.profile.parameters.temperature or 0.7,
            "top_p": self.profile.parameters.top_p or 0.95,
            "top_k": self.profile.parameters.top_k or 40,
            "max_tokens": self.profile.parameters.max_tokens or 4096,
            "stop": self.profile.parameters.stop or stop,
            "repeat_penalty": self.profile.parameters.repeat_penalty or 1.05,
            "stream": stream,
        }
        if self.grammar and not disable_grammar:
            try:  # pragma: no cover
                from llama_cpp import llama_grammar as _llama_grammar  # type: ignore
                completion_kwargs["grammar"] = _llama_grammar.LlamaGrammar.from_json_schema(
                    json.dumps(self.grammar.model_json_schema())
                )
            except Exception as e:  # pragma: no cover
                self._logger.warning(f"Failed to attach grammar on fallback path: {e}")

        llama_ref = getattr(self, "llama_instance", None)
        if llama_ref is None:
            raise RuntimeError("llama_instance missing before completion (vision fallback)")
        return llama_ref.create_completion(**completion_kwargs)


__all__ = ["Qwen3VLPipeline"]
