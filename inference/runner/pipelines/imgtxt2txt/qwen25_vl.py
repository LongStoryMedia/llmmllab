"""
Pipeline for Qwen 2.5 Vision Language GGUF models.
Clean implementation with only essential methods for public API.
"""

import os
import logging
import datetime
from typing import List, Optional, Dict, Any, cast, Type, Union
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from models import (
    ChatResponse,
    Message,
    MessageContent,
    MessageContentType,
    MessageRole,
    Model,
    ModelProfile,
)
from ..base_langgraph import CircuitBreakerConfig
from ..llamacpp.base_llamacpp import BaseLlamaCppPipeline

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

ReturnType = Union[str, ChatResponse]


class Qwen25VLPipeline(BaseLlamaCppPipeline):
    """
    Pipeline class for Qwen 2.5 Vision Language GGUF models using llama-cpp-python.
    Uses the Qwen25VLChatHandler for proper multimodal support.
    Clean implementation with only essential methods.
    """

    # Override allowed return types to include Type for compatibility with typing system
    allowed_return_types: tuple[type, ...] = (str, ChatResponse, list, Type)

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        expected_return_type: Optional[type] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
    ):
        """Initialize a Qwen25VLPipeline instance."""
        # Create logger early so we can use it
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Log the received circuit config for debugging
        if circuit_config is not None:
            self._logger.info(
                f"Qwen25VLPipeline: Received circuit_config with perplexity_guard={circuit_config.enable_perplexity_guard}"
            )
        else:
            self._logger.info(
                "Qwen25VLPipeline: No circuit_config provided, will use defaults from BaseLangGraphPipeline"
            )

        # Let the parent class handle circuit breaker configuration and defaults
        # Initialize with ChatResponse as the expected return type for multimodal
        super().__init__(
            model,
            profile,
            expected_return_type or ChatResponse,
            circuit_config,
        )
        self.model = model
        self.profile = profile
        self.llm: Optional[Llama] = None

        # Validate required model details
        if not (model.details and model.model):
            raise ValueError("Model definition requires model details.")

        # Validate GGUF file
        gguf_path = self._get_gguf_path()
        self._validate_gguf_file(gguf_path)

        self._logger.info(f"Initialized Qwen 2.5 VL GGUF pipeline: {model.name}")

    def _validate_gguf_file(self, gguf_path: str) -> None:
        """Validate that the GGUF file exists and is accessible."""
        # Allow bypassing validation in dev/test environments
        if os.environ.get("ALLOW_MISSING_GGUF", "false").lower() in (
            "1",
            "true",
            "yes",
        ):  # pragma: no cover
            self._logger.warning(
                f"Skipping GGUF validation for dev/test (ALLOW_MISSING_GGUF set). Expected at: {gguf_path}"
            )
            return

        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

        if not os.access(gguf_path, os.R_OK):
            raise PermissionError(f"Cannot read GGUF file: {gguf_path}")

    def _get_gguf_path(self) -> str:
        """Get GGUF file path."""
        return (
            self.model.details.gguf_file
            if self.model.details.gguf_file
            else self.model.model
        )

    def _initialize_llama_cpp_direct(self) -> None:
        """Initialize the Llama model with multimodal support."""
        if self.llm is not None:
            return

        gguf_path = self._get_gguf_path()
        mmproj_path = "/models/qwen2.5-vl-32b-instruct/mmproj-Qwen_Qwen2.5-VL-32B-Instruct-bf16.gguf"

        # Validate file paths
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF model file not found: {gguf_path}")
        if not os.path.exists(mmproj_path):
            raise FileNotFoundError(f"MMProj file not found: {mmproj_path}")

        self._logger.info(f"Loading GGUF model from: {gguf_path}")
        self._logger.info(f"Loading mmproj from: {mmproj_path}")

        # Get circuit breaker configuration for perplexity guard
        enable_perplexity = self.circuit_config.enable_perplexity_guard
        if enable_perplexity is None:
            enable_perplexity = True  # Default to enabled if not specified

        logits_all = enable_perplexity
        logprobs = 1 if enable_perplexity else 0

        self._logger.info(
            f"Perplexity guard {'enabled' if enable_perplexity else 'disabled'} - loading with logits_all={logits_all}, logprobs={logprobs}"
        )

        try:
            chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path)
            self.llm = Llama(
                model_path=gguf_path,
                chat_handler=chat_handler,
                n_gpu_layers=-1,
                n_threads=4,
                verbose=True,
                logits_all=logits_all,  # Respect circuit breaker configuration
                logprobs=logprobs,  # Respect circuit breaker configuration
                embedding=False,
                n_ctx=96000,
                type_k=1,
                type_v=1,
                n_batch=256,
                n_ubatch=128,
                flash_attn=True,
                tensor_split=[0.5, 0.25, 0.25],
                f16_kv=True,
                use_mlock=False,
                use_mmap=True,
                numa=True,
            )
            self._logger.info("Successfully loaded Qwen 2.5 VL model")
        except Exception as e:
            self._logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load Qwen2.5-VL model: {e}") from e

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert internal messages to OpenAI format."""
        formatted_messages = []
        for message in messages:
            role = message.role.value.lower()
            content_list = []

            for content_item in message.content:
                if content_item.type == MessageContentType.TEXT:
                    content_list.append({"type": "text", "text": content_item.text})
                elif content_item.type == MessageContentType.IMAGE:
                    if hasattr(content_item, "url") and content_item.url:
                        content_list.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": content_item.url},
                            }
                        )

            formatted_messages.append({"role": role, "content": content_list})
        return formatted_messages

    def create_graph(
        self, tools: Optional[List[BaseTool]] = None
    ) -> CompiledStateGraph:
        """Create a simple state graph for multimodal processing."""
        # Placeholder - implement if LangGraph integration needed
        raise NotImplementedError(
            "LangGraph integration not implemented for multimodal pipeline"
        )

    def _create_system_prompt(self) -> str:
        """Stub implementation - multimodal pipelines don't use traditional system prompts."""
        return ""
