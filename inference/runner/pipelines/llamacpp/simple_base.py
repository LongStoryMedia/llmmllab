"""
Simplified LlamaCpp base - essential parameter calculation without LangGraph orchestration.
Extracted from complex base_llamacpp.py, preserving original parameter logic.
"""

import os
import logging
from typing import Optional, List

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
from runner.pipelines.base import SimpleChatPipeline, GrammarInput


class BaseLlamaCppPipeline(SimpleChatPipeline):  # pyright: ignore[reportIncompatibleMethodOverride]
    """Unified llama.cpp pipeline base combining simple fast path with full feature support.

    Features merged from legacy advanced base:
    - Heuristic backoff across (n_batch, n_gpu_layers, n_ctx)
    - Perplexity guard toggling logits/logprobs
    - Grammar (GBNF/Pydantic) preparation hooks (stored for later validation)
    - Explicit gpu_layers override + -1 full offload first strategy
    - n_cpu_moe, flash_attention, seed, stop sequence handling
    - Structured, progressive logging of attempts / fallbacks
    """

    def __init__(self, model: Model, profile: ModelProfile):
        super().__init__(model, profile)
        self.llm: Optional[BaseChatModel] = None
        self._logger = logging.getLogger(self.__class__.__name__)
        self._grammar_model_class: Optional[type] = None

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

    def _calculate_optimal_gpu_layers(
        self, n_ctx: int, model_size_category: str
    ) -> int:
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

    # ---------- Optional grammar (store only; actual enforcement delegated elsewhere) ----------
    def _process_grammar_input(self, grammar: Optional[GrammarInput]) -> Optional[str]:
        if grammar is None:
            return None
        try:
            from utils.grammar_generator import (
                get_grammar_for_model,
                load_grammar_from_file,
            )
            from pydantic import BaseModel
            from pathlib import Path
            if isinstance(grammar, str):
                return grammar
            if isinstance(grammar, Path):
                return load_grammar_from_file(grammar)
            if isinstance(grammar, type) and issubclass(grammar, BaseModel):
                self._grammar_model_class = grammar  # store for potential validation
                return get_grammar_for_model(grammar)
        except Exception as e:  # pragma: no cover
            self._logger.warning(f"Grammar processing failed: {e}")
        return None

    def _initialize_llamacpp_with_fallback(
        self,
        gguf_path: str,
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[GrammarInput] = None,
    ) -> BaseChatModel:
        """Full feature initialization with heuristic backoff across context/batch/gpu layers."""

        # Get base parameters from profile
        requested_ctx = self.profile.parameters.num_ctx or 4096
        requested_batch = self.profile.parameters.batch_size or 512

        # Get model size category
        model_size_category = self._get_model_size_category()

        # Build context candidates
        context_candidates = self._build_context_candidates(requested_ctx)

        # Perplexity / logits guard (mirrors advanced base logic simplified)
        perplexity_enabled = bool(
            getattr(self.profile.parameters, "enable_perplexity_guard", False)
        )
        logits_all_enabled = perplexity_enabled
        logprobs = 1 if perplexity_enabled else 0
        if perplexity_enabled:
            self._logger.info(
                "Perplexity guard enabled -> logits_all=True logprobs=1 (memory heavier)"
            )
        else:
            self._logger.info(
                "Perplexity guard disabled -> logits_all=False logprobs=0 (memory optimized)"
            )

        # Pre-compute grammar string (best effort)
        grammar_string = None
        if grammar is not None:
            grammar_string = self._process_grammar_input(grammar)
            if grammar_string:
                self._logger.info(
                    f"Grammar prepared ({len(grammar_string)} chars). Passing if supported."
                )

        for n_ctx in context_candidates:  # outer loop: context last to reduce
            n_batch_base = min(requested_batch, max(256, n_ctx // 64))
            batch_candidates = sorted(
                {n_batch_base, max(256, n_batch_base // 2), 256}, reverse=True
            )

            # Determine GPU layers: explicit override > heuristic
            explicit_gpu_layers: Optional[int] = None
            if (
                self.profile.gpu_config is not None
                and self.profile.gpu_config.gpu_layers is not None
            ):
                explicit_gpu_layers = self.profile.gpu_config.gpu_layers
                self._logger.info(
                    f"Explicit gpu_layers from profile: {explicit_gpu_layers}"
                )
            force_full_env = os.getenv("FORCE_FULL_GPU", "false").lower() in {
                "1",
                "true",
                "yes",
            }

            # Build candidate list: ALWAYS try -1 first unless explicit override is non -1
            gpu_layer_candidates: List[int] = []
            if explicit_gpu_layers is not None:
                if explicit_gpu_layers == -1:
                    gpu_layer_candidates = [-1]
                else:
                    # Explicit numeric override first, then a conservative fallback (2/3) then heuristic
                    conservative = max(1, int(explicit_gpu_layers * 2 / 3))
                    heuristic = self._calculate_optimal_gpu_layers(
                        n_ctx, model_size_category
                    )
                    gpu_layer_candidates = [explicit_gpu_layers]
                    if conservative != explicit_gpu_layers:
                        gpu_layer_candidates.append(conservative)
                    if heuristic not in gpu_layer_candidates:
                        gpu_layer_candidates.append(heuristic)
            else:
                # No explicit override: attempt full offload first (-1), unless user disables via env (they can set DISABLE_FULL_GPU=1 hypothetically)
                disable_full = os.getenv("DISABLE_FULL_GPU", "false").lower() in {
                    "1",
                    "true",
                    "yes",
                }
                heuristic = self._calculate_optimal_gpu_layers(
                    n_ctx, model_size_category
                )
                fallback2 = int(heuristic * 0.9)
                fallback3 = int(heuristic * 0.8)
                if not disable_full or force_full_env:
                    gpu_layer_candidates.append(-1)
                # Add heuristic + fallbacks ensuring uniqueness & positivity
                for cand in [heuristic, fallback2, fallback3]:
                    if cand > 0 and cand not in gpu_layer_candidates:
                        gpu_layer_candidates.append(cand)
                # Ensure a minimal floor candidate
                if 16 not in gpu_layer_candidates:
                    gpu_layer_candidates.append(16)

            # If FORCE_FULL_GPU set and -1 not present (e.g., explicit override path) prepend -1
            if force_full_env and -1 not in gpu_layer_candidates:
                gpu_layer_candidates.insert(0, -1)

            self._logger.info(
                "GPU layer candidate sequence",
                extra={
                    "context": n_ctx,
                    "candidates": gpu_layer_candidates,
                    "model_size": model_size_category,
                },
            )

            # Iterate GPU layer candidates for this context until one works
            for n_batch in batch_candidates:  # inner first axis
                for n_gpu_layers in gpu_layer_candidates:  # second axis
                    try:
                        kwargs = {
                            "model_path": gguf_path,
                            "f16_kv": True,
                            "n_parts": -1,
                            "n_gpu_layers": n_gpu_layers,
                            "n_ctx": n_ctx,
                            "n_batch": n_batch,
                            "use_mmap": True,
                            "use_mlock": False,
                            "seed": self.profile.parameters.seed or -1,
                            "temperature": self.profile.parameters.temperature or 0.7,
                            "max_tokens": self.profile.parameters.max_tokens or 4096,
                            "top_p": self.profile.parameters.top_p or 0.8,
                            "top_k": self.profile.parameters.top_k or 20,
                            "repeat_penalty": self.profile.parameters.repeat_penalty
                            or 1.05,
                            "stop": self.profile.parameters.stop or [],
                            "streaming": True,
                            "verbose": os.getenv("LOG_LEVEL", "WARNING").lower()
                            == "debug",
                            "logits_all": logits_all_enabled,
                            "logprobs": logprobs,
                            "n_cpu_moe": getattr(self.profile.parameters, "n_cpu_moe", 0),
                            "flash_attention": getattr(
                                self.profile.parameters, "flash_attention", True
                            ),
                            "callback_manager": CallbackManager(
                                [StreamingStdOutCallbackHandler()]
                            ),
                        }
                        if grammar_string:
                            kwargs["grammar"] = grammar_string
                        if ChatLlamaCpp is None:
                            raise ImportError(
                                "ChatLlamaCpp not available - langchain_community required"
                            )
                        llm = ChatLlamaCpp(
                            **{k: v for k, v in kwargs.items() if v is not None}
                        )
                        if tools:
                            try:
                                llm = llm.bind_tools(tools)  # type: ignore[assignment]
                            except Exception as tool_err:  # pragma: no cover
                                self._logger.warning(
                                    f"Tool binding failed (gpu_layers={n_gpu_layers}): {tool_err}"
                                )
                        self._logger.info(
                            "Loaded llama.cpp model ctx=%d batch=%d gpu_layers=%d logits_all=%s logprobs=%d",
                            n_ctx,
                            n_batch,
                            n_gpu_layers,
                            logits_all_enabled,
                            logprobs,
                        )
                        if n_gpu_layers == -1:
                            self._logger.info(
                                "All layers scheduled for GPU (-1). Monitor VRAM for confirmation."
                            )
                        else:
                            self._logger.debug(
                                "Partial offload n_gpu_layers=%d (set -1 or profile override for full).",
                                n_gpu_layers,
                            )
                        return llm  # type: ignore[return-value]
                    except Exception as e:  # noqa: BLE001
                        self._logger.warning(
                            "Load attempt failed ctx=%d batch=%d gpu_layers=%d: %s",
                            n_ctx,
                            n_batch,
                            n_gpu_layers,
                            e,
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
        """Initialize LLM using full heuristic backoff."""
        if self.llm is not None:
            return self.llm
        gguf_path = self._get_gguf_path()
        self.llm = self._initialize_llamacpp_with_fallback(gguf_path, tools)
        return self.llm

# Backwards compatibility alias
SimpleLlamaCppPipeline = BaseLlamaCppPipeline

__all__ = ["BaseLlamaCppPipeline", "SimpleLlamaCppPipeline"]
