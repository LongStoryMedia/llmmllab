"""
Qwen3 MoE pipeline as BaseChatModel implementation.
Provides custom model-specific optimizations for Qwen MoE models.
"""

import logging
import re
from typing import List, Optional, Dict, Any

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

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "qwen3-moe-llamacpp"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        base_params = super()._identifying_params
        base_params.update({
            "model_type": "qwen3-moe",
            "chat_format": "chatml",
        })
        return base_params

    def _get_llama_instance(
        self, 
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[GrammarInput] = None
    ) -> Any:
        """Get or create Llama instance with Qwen-specific optimizations."""
        if self.llama_instance is None:
            gguf_path = self._get_gguf_path()
            self.llama_instance = self._initialize_llama_with_qwen_optimizations(
                gguf_path, tools, grammar
            )
                
        return self.llama_instance

    def _initialize_llama_with_qwen_optimizations(
        self,
        gguf_path: str,
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[GrammarInput] = None,
    ) -> Any:
        """Initialize Llama with Qwen-specific optimizations."""
        from llama_cpp import Llama
        
        if Llama is None:
            raise ImportError("llama-cpp-python is required but not installed")

        # Get base parameters
        requested_ctx = self.profile_config.parameters.num_ctx or 32768  # Qwen supports larger contexts
        requested_batch = self.profile_config.parameters.batch_size or 1024  # Larger batch for MoE
        
        # Qwen MoE-specific optimizations
        model_size_category = self._get_model_size_category()
        
        # For MoE models, be more aggressive with GPU layers
        explicit_gpu_layers = None
        if (
            self.profile_config.gpu_config is not None
            and self.profile_config.gpu_config.gpu_layers is not None
        ):
            explicit_gpu_layers = self.profile_config.gpu_config.gpu_layers
        else:
            # Default to full offload for MoE models
            explicit_gpu_layers = -1

        try:
            # Initialize with Qwen-optimized parameters
            llama_params = {
                "model_path": gguf_path,
                "n_gpu_layers": explicit_gpu_layers,
                "n_ctx": requested_ctx,
                "n_batch": requested_batch,
                "n_threads": self._get_optimal_threads(),
                "seed": self.profile_config.parameters.seed or -1,
                "temperature": self.profile_config.parameters.temperature or 0.7,
                "top_p": self.profile_config.parameters.top_p or 0.8,
                "top_k": self.profile_config.parameters.top_k or 20,
                "repeat_penalty": self.profile_config.parameters.repeat_penalty or 1.05,
                "use_mmap": True,
                "use_mlock": False,
                "f16_kv": True,
                "verbose": False,
                "flash_attn": True,  # Enable flash attention for better performance
                "logits_all": False,
                "embedding": False,
                "chat_format": "chatml",  # Qwen uses ChatML format
                # MoE-specific parameters
                "n_cpu_moe": getattr(self.profile_config.parameters, "n_cpu_moe", 0),
            }

            llama_instance = Llama(**llama_params)
            
            self._logger.info(
                f"Initialized Qwen3 MoE model: ctx={requested_ctx}, batch={requested_batch}, "
                f"gpu_layers={explicit_gpu_layers}, chat_format=chatml"
            )
            
            return llama_instance

        except Exception as e:
            self._logger.error(f"Failed to initialize Qwen3 MoE model: {e}")
            # Fallback to base initialization
            return super()._initialize_llama_with_fallback(gguf_path, tools, grammar)

    def _extract_response_content(self, raw_response: str) -> str:
        """Extract response content and handle <think> tags for reasoning models."""
        # Remove <think>...</think> blocks for cleaner output
        cleaned = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL)

        # Clean up extra whitespace
        cleaned = re.sub(r"\n\s*\n", "\n", cleaned)
        cleaned = cleaned.strip()

        return cleaned or raw_response  # Fallback to original if nothing left

    def _format_messages_for_llama(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Override message formatting for Qwen-specific chat format."""
        # Use base implementation but ensure proper ChatML format
        llama_messages = super()._format_messages_for_llama(messages)
        
        # Qwen models work best with ChatML format, which is handled by the base class
        # but we could add Qwen-specific message processing here if needed
        return llama_messages

    def _post_process_response(self, response_content: str) -> str:
        """Post-process response to handle Qwen-specific patterns."""
        # Extract clean content, removing think tags
        return self._extract_response_content(response_content)


__all__ = ["Qwen3Moe"]