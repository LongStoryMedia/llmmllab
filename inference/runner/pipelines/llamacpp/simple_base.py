"""
Simplified LlamaCpp base - essential parameter calculation without LangGraph orchestration.
Extracted from complex base_llamacpp.py, preserving original parameter logic.
"""

import os
import logging
from typing import Optional, List
import torch

try:
    from langchain_community.chat_models.llamacpp import ChatLlamaCpp
except ImportError:
    try:
        from langchain.llms.llamacpp import LlamaCpp as ChatLlamaCpp  # type: ignore[import]
    except ImportError:  # pragma: no cover
        ChatLlamaCpp = None
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from models import Model, ModelProfile
from .utils import calculate_optimal_gpu_layers  # Reuse aggressive heuristic
from runner.pipelines.base import SimpleChatPipeline


class SimpleLlamaCppPipeline(SimpleChatPipeline):
    """
    Simplified LlamaCpp pipeline base with original parameter calculation.

    Preserves the essential parameter calculation logic from BaseLlamaCppPipeline
    but removes LangGraph orchestration complexity.
    """

    def __init__(self, model: Model, profile: ModelProfile):
        super().__init__(model, profile)
        self.llm: Optional[BaseChatModel] = None
        self._logger = logging.getLogger(self.__class__.__name__)

    def _get_gguf_path(self) -> str:
        """Get the GGUF file path from model definition."""
        assert self.model
        return (
            self.model.details.gguf_file
            if hasattr(self.model.details, "gguf_file") and self.model.details.gguf_file
            else self.model.model
        )

    def _get_optimal_threads(self) -> int:
        """Get optimal thread count based on system capabilities."""
        try:
            import multiprocessing

            cpu_count = multiprocessing.cpu_count()
            optimal_threads = min(max(cpu_count // 2, 2), 8)
            self._logger.debug(
                f"Using {optimal_threads} threads (CPU count: {cpu_count})"
            )
            return optimal_threads
        except Exception:
            self._logger.warning(
                "Could not determine CPU count, using default threading"
            )
            return 4

    def _calculate_optimal_gpu_layers(self, n_ctx: int, model_size_category: str) -> int:
        """Delegate to shared aggressive heuristic for consistency with advanced pipelines."""
        return calculate_optimal_gpu_layers(n_ctx, model_size_category)

    def _get_model_size_category(self) -> str:
        """Determine model size category from model name."""
        model_name = self.model.name.lower()

        # Extract parameter count from common naming patterns
        if any(x in model_name for x in ["1b", "1.5b", "3b"]):
            return "small"
        elif any(x in model_name for x in ["7b", "8b", "13b"]):
            return "medium"
        elif any(x in model_name for x in ["20b", "30b"]):
            return "large"
        elif any(x in model_name for x in ["70b", "120b"]):
            return "xlarge"
        else:
            # Default based on typical usage
            return "medium"

    def _build_context_candidates(self, requested_ctx: int) -> List[int]:
        """Build context size candidates based on original logic."""
        # Original logic from base_llamacpp.py
        candidates = []

        # Add the requested context
        candidates.append(requested_ctx)

        # Add some fallback contexts
        fallbacks = [32768, 16384, 8192, 4096, 2048]
        for fallback in fallbacks:
            if fallback < requested_ctx and fallback not in candidates:
                candidates.append(fallback)

        # Sort in descending order (try largest first)
        return sorted(list(set(candidates)), reverse=True)

    def _initialize_llamacpp_with_fallback(
        self,
        gguf_path: str,
        tools: Optional[List[BaseTool]] = None,
    ) -> BaseChatModel:
        """Initialize ChatLlamaCpp with simplified fallback logic."""

        # Get base parameters from profile
        requested_ctx = self.profile.parameters.num_ctx or 4096
        requested_batch = self.profile.parameters.batch_size or 512

        # Get model size category
        model_size_category = self._get_model_size_category()

        # Build context candidates
        context_candidates = self._build_context_candidates(requested_ctx)

        # Try context sizes in order
        for n_ctx in context_candidates:
            # Calculate batch size (original logic)
            n_batch = min(requested_batch, max(256, n_ctx // 64))

            # Determine GPU layers: explicit override > heuristic
            explicit_gpu_layers = None
            if getattr(self.profile, "gpu_config", None) and getattr(getattr(self.profile, "gpu_config"), "gpu_layers", None) is not None:
                explicit_gpu_layers = getattr(self.profile.gpu_config, "gpu_layers")
                self._logger.info(
                    f"Using explicit gpu_layers override from profile: {explicit_gpu_layers}"
                )

            force_full_env = os.getenv("FORCE_FULL_GPU", "false").lower() in {"1", "true", "yes"}

            if explicit_gpu_layers is not None:
                # Honor sentinel -1 (all layers) explicitly, never mutate
                if explicit_gpu_layers == -1:
                    self._logger.info(
                        "Full GPU offload requested via profile (gpu_layers=-1). Passing through to llama.cpp."
                    )
                    n_gpu_layers = -1
                else:
                    n_gpu_layers = explicit_gpu_layers
            elif force_full_env:
                self._logger.info(
                    "FORCE_FULL_GPU environment variable set – using n_gpu_layers=-1 (all layers)."
                )
                n_gpu_layers = -1
            else:
                # Use aggressive heuristic, then clamp to at least 1. If heuristic returns > 0 we keep it.
                heuristic_layers = self._calculate_optimal_gpu_layers(n_ctx, model_size_category)
                n_gpu_layers = heuristic_layers if heuristic_layers > 0 else 1
                # Warn if heuristic result seems suspiciously low (e.g., < 24 for medium/large models at modest context)
                if n_gpu_layers < 24 and model_size_category in {"medium", "large", "xlarge"} and n_ctx <= 32768:
                    self._logger.warning(
                        "Heuristic GPU layer allocation appears low (n_gpu_layers=%d, ctx=%d, size=%s). "
                        "Consider setting gpu_layers=-1 in profile or FORCE_FULL_GPU=1 for full offload.",
                        n_gpu_layers,
                        n_ctx,
                        model_size_category,
                    )

            try:
                # Use original parameter structure
                kwargs = {
                    "model_path": gguf_path,
                    "n_ctx": n_ctx,
                    "n_gpu_layers": n_gpu_layers,
                    "n_threads": self._get_optimal_threads(),
                    "f16_kv": True,
                    "verbose": os.getenv("LOG_LEVEL", "WARNING").lower() == "debug",
                    "n_batch": n_batch,
                    "n_parts": -1,
                    "seed": self.profile.parameters.seed or -1,
                    "logits_all": False,  # Simplified: no perplexity guard
                    "vocab_only": False,
                    "use_mlock": False,
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    # Add streaming parameters
                    "streaming": True,
                    "callback_manager": CallbackManager(
                        [StreamingStdOutCallbackHandler()]
                    ),
                    # Add generation parameters from profile
                    "temperature": self.profile.parameters.temperature or 0.7,
                    "max_tokens": self.profile.parameters.max_tokens or 4096,
                    "top_p": self.profile.parameters.top_p or 0.8,
                    "top_k": self.profile.parameters.top_k or 20,
                    "repeat_penalty": self.profile.parameters.repeat_penalty or 1.05,
                    "stop": self.profile.parameters.stop or [],
                }

                # Try to initialize
                if ChatLlamaCpp is None:
                    raise ImportError(
                        "ChatLlamaCpp not available - langchain_community required"
                    )
                llm = ChatLlamaCpp(**kwargs)

                # Store the base LLM
                base_llm = llm

                # Bind tools if provided (this changes the type, so handle separately)
                if tools:
                    try:
                        llm.bind_tools(tools)
                    except Exception as tool_err:  # pragma: no cover
                        self._logger.warning(f"Tool binding failed: {tool_err}")

                self._logger.info(
                    f"Loaded {self.model.name} successfully: "
                    f"ctx={n_ctx}, batch={n_batch}, gpu_layers={n_gpu_layers}"
                )
                if n_gpu_layers == -1:
                    self._logger.info(
                        "All model layers scheduled for GPU. Monitor VRAM to ensure allocation succeeded."
                    )
                else:
                    self._logger.info(
                        "Partial layer offload (n_gpu_layers=%d). Set gpu_layers=-1 or FORCE_FULL_GPU=1 for full offload.",
                        n_gpu_layers,
                    )

                return base_llm

            except Exception as e:
                self._logger.warning(
                    f"Failed to load with ctx={n_ctx}, batch={n_batch}, gpu_layers={n_gpu_layers}: {e}"
                )
                continue

        # If all attempts failed, raise the last error
        raise RuntimeError(
            f"Failed to initialize {self.model.name} with any configuration"
        )

    def _initialize_llm(
        self,
        tools: Optional[List[BaseTool]] = None,
    ) -> BaseChatModel:
        """Initialize LLM using original parameter calculation logic."""
        if self.llm is not None:
            return self.llm

        gguf_path = self._get_gguf_path()
        self.llm = self._initialize_llamacpp_with_fallback(gguf_path, tools)
        return self.llm
