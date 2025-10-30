"""
Qwen3 VL pipeline for multimodal (text + image) generation.
Optimized for Qwen3 VL models with vision capabilities.
"""

import os
import json
from typing import Dict, Any, Optional, Type, List
from pydantic import BaseModel  # noqa: F401
# llama_cpp imported lazily within methods to reduce unnecessary top-level dependencies
try:
    from PIL import Image  # type: ignore
except Exception:  # Pillow may not be installed in minimal environment
    Image = None  # type: ignore

from models import Model, ModelProfile
from runner.pipelines.llamacpp import BaseLlamaCppPipeline


class Qwen3VLPipeline(BaseLlamaCppPipeline):
    """
    Qwen3 VL multimodal chat model implementation.

    Features:
    - Optimized for Qwen3 VL models (e.g., Qwen3-VL-32B-Thinking-abliterated)
    - Vision capabilities with multimodal processing
    - Custom chat format for Qwen3 VL models
    - Hardware optimization for large VL models
    - <think> tag processing for reasoning models
    - Supports image and video inputs
    """

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        grammar: Optional[Type[BaseModel]] = None,
        **kwargs
    ):
        # Store multimodal-specific parameters before calling super().__init__
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

    def _get_clip_model_path(self) -> Optional[str]:
        """Get the clip model path for multimodal processing from model details.

        Uses getattr to avoid static typing errors if attribute not declared in ModelDetails.
        """
        details = getattr(self.model, "details", None)
        if details is not None:
            clip_path = getattr(details, "clip_model_path", None)
            if clip_path:
                if os.path.exists(clip_path):
                    self._logger.info(f"Using clip model from config: {clip_path}")
                    return clip_path
                else:
                    self._logger.warning(
                        f"Configured clip model path does not exist: {clip_path}"
                    )
        self._logger.warning(
            f"No clip model path configured for {self.model.name}"
        )
        return None

    def _create_chat_handler(self):
        """Create the appropriate chat handler for Qwen3 VL multimodal processing."""
        try:
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler
            clip_path = self._get_clip_model_path()
            if clip_path:
                self._logger.info(f"Creating Qwen25VLChatHandler with clip model: {clip_path}")
                # Create the chat handler with the clip model path
                return Qwen25VLChatHandler(clip_model_path=clip_path)
            else:
                self._logger.error("No clip model path available for multimodal chat handler")
                return None
        except ImportError:
            self._logger.error("Qwen25VLChatHandler not available in this llama-cpp-python version")
            return None
        except Exception as e:
            self._logger.error(f"Failed to create Qwen25VLChatHandler: {e}")
            return None



    def _get_chat_handler(self):
        """Override to provide multimodal chat handler."""
        return self._create_chat_handler()

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

    def _load_images(self, paths: List[str]) -> List[Any]:
        images: List[Any] = []
        if not paths and Image is not None:
            # Provide a placeholder 1024x1024 black image to satisfy handler when no images are passed.
            try:
                placeholder = Image.new("RGB", (1024, 1024), color=(0, 0, 0))
                images.append(placeholder)
                self._logger.warning(
                    "No images detected; supplied placeholder to avoid vision handler segfault"
                )
                return images
            except Exception as e:
                self._logger.error(f"Failed to create placeholder image: {e}")
                return images
        if Image is None:
            if paths:
                self._logger.error(
                    "Pillow not available; cannot load image paths required for multimodal request"
                )
            return images
        for p in paths:
            try:
                if os.path.exists(p):
                    img = Image.open(p)
                    images.append(img)
                else:
                    self._logger.warning(f"Image path does not exist: {p}")
            except Exception as e:
                self._logger.error(f"Failed to load image '{p}': {e}")
        if not images and paths:
            self._logger.warning(
                "No valid images loaded; providing placeholder to avoid segfault"
            )
            try:
                placeholder = Image.new("RGB", (1024, 1024), color=(0, 0, 0))
                images.append(placeholder)
            except Exception:
                pass
        return images

    def _get_res(
        self,
        messages,
        stop: Optional[List[str]] = None,
        tools: Optional[List[Any]] = None,
        stream: bool = False,
    ):
        """Extend base response retrieval to include images for multimodal."""
        image_paths = self._extract_image_paths(messages)
        images = self._load_images(image_paths)
        # Delegate to base logic for text/tool formatting
        converted_tools = self._convert_tools_to_simple_format(tools)
        llama_messages = self._format_messages_for_llama(messages)
        self._logger.info(
            f"Chat completion (VL): model={self.model.name}, messages={len(llama_messages)}, tools={len(converted_tools) if converted_tools else 0}, images={len(images)}"
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
        if images:
            kwargs["images"] = images
        if not getattr(self, "llama_instance", None):
            raise RuntimeError("Llama instance not initialized for Qwen3VLPipeline")
        return self.llama_instance.create_chat_completion(**kwargs)  # type: ignore


__all__ = ["Qwen3VLPipeline"]