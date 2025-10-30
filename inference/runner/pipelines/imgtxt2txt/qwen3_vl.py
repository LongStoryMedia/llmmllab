"""
Qwen3 VL pipeline for multimodal (text + image) generation.
Optimized for Qwen3 VL models with vision capabilities.
"""

import os
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel
import llama_cpp

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
        """Get the clip model path for multimodal processing from model details."""
        if hasattr(self.model.details, 'clip_model_path') and self.model.details.clip_model_path:
            clip_path = self.model.details.clip_model_path
            if os.path.exists(clip_path):
                self._logger.info(f"Using clip model from config: {clip_path}")
                return clip_path
            else:
                self._logger.warning(f"Configured clip model path does not exist: {clip_path}")
        
        self._logger.warning(f"No clip model path configured for {self.model.name}")
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

    def _initialize_llama_simple(self, gguf_path: str):
        """Override to add multimodal chat handler support."""
        # Create the chat handler before initializing
        self._multimodal_chat_handler = self._create_chat_handler()
        
        # Call parent method but we'll need to modify the llama creation
        return self._initialize_llama_simple_with_multimodal(gguf_path)

    def _initialize_llama_simple_with_multimodal(self, gguf_path: str):
        """Initialize llama with multimodal chat handler."""
        from runner.utils.hardware_manager import hardware_manager
        
        self._logger.info(
            f"🚀 Simple multimodal initialization {self.model.name}: "
            f"n_ctx={self.profile.parameters.num_ctx or 4096}, "
            f"n_batch={getattr(self.profile.parameters, 'batch_size', None) or 64}, "
            f"gpu_layers=-1, multimodal={'yes' if self._multimodal_chat_handler else 'no'}"
        )

        try:
            # Use the same parameters as the parent but add chat_handler
            llama_instance = llama_cpp.Llama(
                model_path=gguf_path,
                n_gpu_layers=-1,  # Full GPU offload by default
                n_ctx=self.profile.parameters.num_ctx or 4096,
                n_batch=getattr(self.profile.parameters, 'batch_size', None) or 64,
                temperature=self.profile.parameters.temperature or 0.7,
                top_p=self.profile.parameters.top_p or 0.8,
                top_k=self.profile.parameters.top_k or 20,
                repeat_penalty=self.profile.parameters.repeat_penalty or 1.05,
                seed=self.profile.parameters.seed or llama_cpp.LLAMA_DEFAULT_SEED,
                chat_handler=self._multimodal_chat_handler,  # Add multimodal support
                verbose=os.getenv("LOG_LEVEL", "WARNING").lower() == "trace",
                flash_attn=getattr(self.profile.parameters, "flash_attention", True),
                use_mmap=True,
                use_mlock=False,
                f16_kv=True,
            )
            
            self._logger.info(f"✅ Multimodal llama instance created successfully for {self.model.name}")
            return llama_instance
            
        except Exception as e:
            self._logger.error(f"❌ Simple multimodal initialization failed for {self.model.name}: {e}")
            raise RuntimeError(f"Failed to initialize {self.model.name}: {e}. Enable ENABLE_INTELLIGENT_OOM_RECOVERY=true for advanced recovery.")

    def _initialize_llama_with_intelligent_oom_recovery(self, gguf_path: str):
        """Override to add multimodal support with OOM recovery."""
        # Create the chat handler before initializing
        self._multimodal_chat_handler = self._create_chat_handler()
        
        # For now, fall back to simple initialization with multimodal support
        # TODO: Integrate with the full OOM recovery system if needed
        return self._initialize_llama_simple_with_multimodal(gguf_path)


__all__ = ["Qwen3VLPipeline"]