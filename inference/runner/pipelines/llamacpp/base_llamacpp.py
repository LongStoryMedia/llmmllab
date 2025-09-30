"""Common llama.cpp pipeline base with heuristic auto-backoff model loading.

Centralizes llama.cpp specific initialization so individual model pipelines
only need to implement prompt/system logic while benefiting from:

* Heuristic auto-backoff loop varying (n_batch, n_gpu_layers, n_ctx) in that priority order
* Simplified feature profile (no progressive logprobs decrement)
* Configurable context candidates derived from requested + model maxima
* Structured logging of each attempt and final chosen configuration
"""

from __future__ import annotations

import os
import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel

from langchain_community.chat_models.llamacpp import ChatLlamaCpp
from langchain_core.callbacks import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from .utils import calculate_optimal_gpu_layers

from runner.pipelines.base_langgraph import (
    BaseLangGraphPipeline,
    CircuitBreakerConfig,
)
from models import Model, ModelProfile
from runner.pipelines.base import GrammarInput


@dataclass
class LlamaLoadAttempt:
    n_ctx: int
    n_batch: int
    n_gpu_layers: int
    logits_all: bool
    logprobs: int
    attempt_index: int
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:  # For structured logging / telemetry
        return {
            "n_ctx": self.n_ctx,
            "n_batch": self.n_batch,
            "n_gpu_layers": self.n_gpu_layers,
            "logits_all": self.logits_all,
            "logprobs": self.logprobs,
            "error": self.error,
            "attempt": self.attempt_index,
        }


class BaseLlamaCppPipeline(BaseLangGraphPipeline):
    """Base class encapsulating llama.cpp loading heuristics.

    Subclasses MUST implement:
        _get_gguf_path()
        _create_system_prompt()
    """

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        expected_return_type: Optional[type] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        model_size_category: str = "large",
    ):
        super().__init__(model, profile, expected_return_type, circuit_config)
        self._logger = logging.getLogger(f\"{__name__}.{self.__class__.__name__}\")\n        self._model_size_category = model_size_category\n        self._grammar_constraint: Optional[str] = None  # Store grammar constraint for validation\n        self._grammar_model_class: Optional[type] = None  # Store Pydantic model class for validation

    @abstractmethod
    def _get_gguf_path(self) -> str:
        """Get the path to the GGUF model file. Must be implemented by subclasses."""
        ...

    @abstractmethod
    async def _create_system_prompt(
        self, tools: Optional[List[BaseTool]] = None
    ) -> str:
        """Create system prompt for the model. Must be implemented by subclasses."""
        ...

    def _get_gpu_config_kwargs(self) -> Dict[str, Any]:
        """Extract GPU configuration from model parameters and convert to llama-cpp-python kwargs.

        Returns:
            Dict of kwargs to pass to ChatLlamaCpp for GPU configuration
        """
        kwargs = {}

        # Get GPU config from profile (separate from parameters now)
        gpu_config = getattr(self.profile, "gpu_config", None)
        if not gpu_config:
            return kwargs

        # Import hardware manager for device name resolution
        from utils.hardware_manager import hardware_manager

        # Main GPU device selection
        if gpu_config.main_gpu is not None and gpu_config.main_gpu >= 0:
            kwargs["main_gpu"] = gpu_config.main_gpu
        elif gpu_config.main_gpu_device_id:
            # Resolve device name to index
            resolved_index = hardware_manager.resolve_device_name_to_index(
                gpu_config.main_gpu_device_id
            )
            if resolved_index >= 0:
                kwargs["main_gpu"] = resolved_index

        # Tensor split configuration
        if gpu_config.tensor_split:
            if gpu_config.tensor_split_devices:
                # Map device names to indices and create proper tensor split
                device_indices = []
                for device_name in gpu_config.tensor_split_devices:
                    device_idx = hardware_manager.resolve_device_name_to_index(
                        device_name
                    )
                    device_indices.append(device_idx)

                # Create tensor split array aligned with device indices
                # Note: llama-cpp-python expects tensor_split as list of floats for each GPU
                max_gpu = max(device_indices) if device_indices else 0
                tensor_split_array = [0.0] * (max_gpu + 1)

                for device_idx, split_value in zip(
                    device_indices, gpu_config.tensor_split
                ):
                    if device_idx >= 0:  # Skip CPU
                        tensor_split_array[device_idx] = split_value

                kwargs["tensor_split"] = tensor_split_array
            else:
                # Use tensor split directly if no device mapping specified
                kwargs["tensor_split"] = gpu_config.tensor_split

        # KV cache offloading
        if gpu_config.no_kv_offload is not None:
            # ChatLlamaCpp may use different parameter names for this
            # Check the actual implementation for correct parameter name
            kwargs["offload_kqv"] = not gpu_config.no_kv_offload
        elif gpu_config.offload_kqv is not None:
            kwargs["offload_kqv"] = gpu_config.offload_kqv

        self._logger.info(f"Applied GPU configuration: {kwargs}")
        return kwargs

    def _process_grammar_input(self, grammar: Optional[GrammarInput]) -> Optional[str]:
        """Process grammar input and return GBNF grammar string.
        
        Args:
            grammar: Grammar input (GBNF string, file path, or Pydantic model class)
            
        Returns:
            GBNF grammar string or None if no grammar provided
        """
        if grammar is None:
            return None
            
        try:
            from utils.grammar_generator import get_grammar_for_model, load_grammar_from_file
            from pydantic import BaseModel
            from pathlib import Path
            
            if isinstance(grammar, str):
                # Assume it's already a GBNF grammar string
                return grammar
            elif isinstance(grammar, Path):
                # Load from file
                return load_grammar_from_file(grammar)
            elif isinstance(grammar, type) and issubclass(grammar, BaseModel):
                # Generate from Pydantic model
                # Store the model class for validation
                self._grammar_model_class = grammar
                return get_grammar_for_model(grammar)
            else:
                self._logger.warning(f"Unsupported grammar type: {type(grammar)}")
                return None
                
        except Exception as e:
            self._logger.error(f"Error processing grammar input: {e}")
            return None

    def validate_output_against_grammar(self, output: str) -> tuple[bool, str]:
        """Validate LLM output against grammar constraint.
        
        Args:
            output: Raw LLM output text
            
        Returns:
            Tuple of (is_valid, validation_message)
        """
        if not hasattr(self, '_grammar_constraint') or not self._grammar_constraint:
            return True, "No grammar constraint"
            
        try:
            # If we have a Pydantic model class, try to parse the output
            if hasattr(self, '_grammar_model_class') and self._grammar_model_class:
                from utils.grammar_generator import parse_structured_output
                try:
                    parsed = parse_structured_output(output, self._grammar_model_class)
                    return True, f"Valid {self._grammar_model_class.__name__}"
                except Exception as e:
                    return False, f"Grammar validation failed: {e}"
            
            # For raw grammar strings, we can't easily validate without a full parser
            # Return True for now, but log that validation is limited
            self._logger.debug("Grammar validation limited for raw GBNF strings")
            return True, "Grammar validation not implemented for raw GBNF"
            
        except Exception as e:
            self._logger.error(f"Error validating grammar: {e}")
            return False, f"Validation error: {e}"

    # ---------- LLM Initialization (Heuristic Backoff) ----------
    async def _initialize_llm(
        self, 
        gguf_path: str, 
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[GrammarInput] = None,
    ) -> None:  # noqa: D401
        """Initialize llama.cpp model with heuristic auto-backoff.

        Strategy order (OOM/backoff preference):
            1. Keep context size (n_ctx) constant while reducing n_batch first
            2. Then reduce number of GPU layers (but never below 2/3 of initial allocation)
            3. Only if all batch + gpu layer combinations fail, drop to the next smaller n_ctx
        We do NOT decrement logprobs progressively anymore (removed as ineffective).
        Stops at first successful load.
        
        Args:
            gguf_path: Path to the GGUF model file
            tools: Optional tools for the pipeline
            grammar: Optional grammar constraint (GBNF string, file path, or Pydantic model class)
        """
        if os.environ.get("ALLOW_MISSING_GGUF", "false").lower() in {
            "1",
            "true",
            "yes",
        }:

            class _DummyLLM(BaseChatModel):  # pragma: no cover - dev convenience
                def bind_tools(self, *_, **__):  # type: ignore[override]
                    return self

                def _stream(self, *_, **__):  # type: ignore[override]
                    """Required by BaseChatModel interface."""
                    sample = ["Dummy", " response", " for", " dev/test", "."]
                    for tok in sample:
                        yield AIMessage(
                            content=tok,
                            response_metadata={
                                "logprobs": {
                                    "content": [{"token": tok, "logprob": -1.0}]
                                }
                            },
                        )

                async def astream(self, *_, **__):  # type: ignore[override]
                    sample = ["Dummy", " response", " for", " dev/test", "."]
                    for tok in sample:
                        yield AIMessage(
                            content=tok,
                            response_metadata={
                                "logprobs": {
                                    "content": [{"token": tok, "logprob": -1.0}]
                                }
                            },
                        )

                async def ainvoke(self, *_, **__):  # type: ignore[override]
                    return AIMessage(content="Dummy response for dev/test.")

                def _generate(self, *_, **__):  # type: ignore[override]
                    return AIMessage(content="Dummy response for dev/test.")

                @property
                def _llm_type(self) -> str:  # noqa: D401
                    return "dummy"

            self.llm = _DummyLLM()
            if tools:
                self.llm = self.llm.bind_tools(tools)
            self._logger.warning("Loaded dummy LLM (ALLOW_MISSING_GGUF enabled)")
            return

        if ChatLlamaCpp is None:  # pragma: no cover
            raise RuntimeError("ChatLlamaCpp not available - install dependencies.")

        requested_ctx = self.profile.parameters.num_ctx or 1048576
        context_candidates = self._build_context_candidates(requested_ctx)
        requested_batch = self.profile.parameters.batch_size or 2048

        attempts: List[LlamaLoadAttempt] = []
        attempt_index = 0

        from utils.hardware_manager import hardware_manager
        from runner.pipeline_factory import (
            pipeline_factory,
        )  # local import to avoid cycle at module import

        for n_ctx in context_candidates:  # Reduce context LAST (outer loop)
            attempts_for_ctx = 0
            max_attempts_this_ctx = 8

            # 1. Batch sizes (primary reduction axis)
            base_batch = min(requested_batch, max(256, n_ctx // 64))
            batch_candidates = self._descending_unique(
                [base_batch, base_batch // 2, base_batch // 4, 256]
            )

            # 2. GPU layer allocation (secondary reduction axis)
            # Always start with -1 (all layers on GPU) for the first attempt
            base_gpu_layers = calculate_optimal_gpu_layers(
                n_ctx, self._model_size_category
            )
            min_gpu_layers = max(1, int(base_gpu_layers * 2 / 3))  # never below 2/3

            # Start with -1 (all GPU layers) for optimal performance, then fallback
            raw_gpu_candidates = [
                -1,  # First attempt: all layers on GPU
                base_gpu_layers,
                int(base_gpu_layers * 0.9),
                int(base_gpu_layers * 0.8),
                int(base_gpu_layers * 0.75),
                int(base_gpu_layers * 0.7),
            ]
            # Handle -1 (all GPU layers) separately since _descending_unique filters it out
            gpu_layer_candidates = [-1]  # Always start with all GPU layers

            # Add the rest of the candidates in descending order
            other_candidates = self._descending_unique(
                raw_gpu_candidates[1:]
            )  # Skip -1
            for g in other_candidates:
                if g >= min_gpu_layers:
                    gpu_layer_candidates.append(g)

            if len(gpu_layer_candidates) == 1:  # Only -1 in the list
                gpu_layer_candidates = [min_gpu_layers]

            # Conditional feature settings based on perplexity guard
            # Only enable logits collection and logprobs if perplexity monitoring is enabled
            perplexity_enabled = self.circuit_config.enable_perplexity_guard or False
            logits_all = perplexity_enabled  # Only collect logits if perplexity monitoring is needed
            logprobs = (
                1 if perplexity_enabled else 0
            )  # Only compute logprobs if needed for perplexity

            if perplexity_enabled:
                self._logger.info(
                    "Perplexity guard enabled - loading with logits_all=True, logprobs=1"
                )
            else:
                self._logger.info(
                    "Perplexity guard disabled - optimizing memory usage with logits_all=False, logprobs=0"
                )

            early_reduce_ctx = False
            for n_batch in batch_candidates:
                for n_gpu_layers in gpu_layer_candidates:
                    attempt_index += 1
                    attempt = LlamaLoadAttempt(
                        n_ctx=n_ctx,
                        n_batch=n_batch,
                        n_gpu_layers=n_gpu_layers,
                        logits_all=logits_all,
                        logprobs=logprobs,
                        attempt_index=attempt_index,
                    )
                    self._logger.info(
                        "llama.cpp load attempt %d: ctx=%d batch=%d gpu_layers=%d logits_all=%s logprobs=%d",
                        attempt_index,
                        n_ctx,
                        n_batch,
                        n_gpu_layers,
                        logits_all,
                        logprobs,
                    )
                    try:
                        # Get GPU configuration kwargs
                        gpu_kwargs = self._get_gpu_config_kwargs()
                        
                        # Process grammar input if provided
                        grammar_kwargs = {}
                        grammar_string = None
                        if grammar is not None:
                            grammar_string = self._process_grammar_input(grammar)
                            if grammar_string:
                                # Store grammar for potential post-processing
                                # Note: Current ChatLlamaCpp may not support grammar directly
                                # We'll store it as metadata for now and implement constraint logic later
                                self._grammar_constraint = grammar_string
                                self._logger.info(f"Grammar constraint prepared: {len(grammar_string)} chars")
                                # Try to set grammar parameter if supported
                                try:
                                    grammar_kwargs["grammar"] = grammar_string
                                except Exception as e:
                                    self._logger.debug(f"Grammar parameter not supported, storing for post-processing: {e}")

                        # Base kwargs for ChatLlamaCpp
                        base_kwargs = {
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
                            "stop": self.profile.parameters.stop
                            or ["<|im_end|>", "<|endoftext|>", "<|end|>"],
                            "streaming": True,
                            "verbose": False,
                            "logprobs": logprobs,
                            "logits_all": logits_all,
                            "n_cpu_moe": self.profile.parameters.n_cpu_moe or 0,
                            "flash_attention": getattr(
                                self.profile.parameters, "flash_attention", True
                            ),
                            "callback_manager": CallbackManager(
                                [StreamingStdOutCallbackHandler()]
                            ),
                        }

                        # Merge GPU configuration kwargs and grammar kwargs, letting GPU config override base settings
                        final_kwargs = {**base_kwargs, **gpu_kwargs, **grammar_kwargs}

                        # Remove any None values and prepare kwargs for ChatLlamaCpp
                        clean_kwargs = {
                            k: v for k, v in final_kwargs.items() if v is not None
                        }

                        self._logger.info(
                            f"Initializing ChatLlamaCpp with: {clean_kwargs}"
                        )

                        # Try to initialize with all parameters, fall back if some are unsupported
                        try:
                            self.llm = ChatLlamaCpp(**clean_kwargs)
                        except TypeError as e:
                            if "unexpected keyword argument" in str(e):
                                # Extract the unsupported parameter from error message
                                error_str = str(e)
                                unsupported_params = []
                                
                                # Check for known potentially unsupported parameters
                                if "n_cpu_moe" in error_str:
                                    unsupported_params.append("n_cpu_moe")
                                if "grammar" in error_str:
                                    unsupported_params.append("grammar")
                                    self._logger.info("Grammar parameter not supported by ChatLlamaCpp, will use validation approach")
                                
                                if unsupported_params:
                                    self._logger.warning(
                                        f"Parameters {unsupported_params} not supported by current version, removing..."
                                    )
                                    clean_kwargs_fallback = {
                                        k: v
                                        for k, v in clean_kwargs.items()
                                        if k not in unsupported_params
                                    }
                                    self.llm = ChatLlamaCpp(**clean_kwargs_fallback)
                                else:
                                    # Re-raise if it's a different unsupported parameter
                                    raise
                            else:
                                # Re-raise if it's not a parameter issue
                                raise
                        if tools:
                            self.llm = self.llm.bind_tools(tools)
                        self._logger.info(
                            "Loaded llama.cpp model ctx=%d batch=%d gpu_layers=%d logits_all=%s logprobs=%d",
                            n_ctx,
                            n_batch,
                            n_gpu_layers,
                            logits_all,
                            logprobs,
                        )
                        # Summarize attempts if any failures preceded
                        failures = [a for a in attempts if a.error]
                        if failures:
                            summary = ", ".join(
                                f"#{a.attempt_index}[ctx={a.n_ctx} batch={a.n_batch} gpu={a.n_gpu_layers} -> {a.error}]"
                                for a in failures
                            )
                            self._logger.info(
                                "Previous failed attempts (most recent last): %s",
                                summary,
                            )
                        return
                    except Exception as e:  # noqa: BLE001
                        attempt.error = str(e)
                        attempts.append(attempt)
                        self._logger.warning(
                            "Load attempt %d failed: %s. Trying next configuration.",
                            attempt_index,
                            e,
                        )

                        # Only perform memory cleanup if this appears to be a memory-related error
                        from utils.hardware_manager import is_memory_related_error

                        # Always clean up the current model instance
                        self.llm = None

                        if is_memory_related_error(e):
                            self._logger.info(
                                "Memory-related error detected, performing cleanup"
                            )
                            try:
                                pipeline_factory.force_resource_cleanup()
                            except Exception:
                                pass
                            hardware_manager.clear_memory(aggressive=True)
                        else:
                            # For non-memory errors (file loading, model format, etc.),
                            # just do basic cleanup without aggressive memory operations
                            self._logger.debug(
                                "Non-memory error detected, skipping aggressive cleanup"
                            )
                            import gc

                            gc.collect()  # Basic garbage collection only
                    finally:
                        attempts_for_ctx += 1
                        if (
                            attempts_for_ctx >= max_attempts_this_ctx
                            and self.llm is None
                        ):
                            self._logger.info(
                                "Reached attempt cap (%d) for ctx=%d; reducing context earlier.",
                                max_attempts_this_ctx,
                                n_ctx,
                            )
                            early_reduce_ctx = True
                if early_reduce_ctx or self.llm is not None:
                    break  # break batch loop
            if early_reduce_ctx and self.llm is None:
                continue  # go to next smaller context

        # Exhausted all attempts
        if attempts:
            summary = "; ".join(
                f"ctx={a.n_ctx} batch={a.n_batch} gpu={a.n_gpu_layers} logits_all={a.logits_all} logprobs={a.logprobs} err={a.error}"  # noqa: E501
                for a in attempts
            )
            self._logger.error(
                "All llama.cpp load attempts failed. Summary: %s", summary
            )
        error_msg = "Failed to load llama.cpp model after heuristic backoff attempts"
        self._mark_model_corrupted(error_msg)
        raise RuntimeError(error_msg)

    # ---------- Helpers ----------

    def _build_context_candidates(self, requested_ctx: int) -> List[int]:
        ladder = [requested_ctx, 524288, 262144, 131072, 65536, 32768, 16384]
        ladder.sort(reverse=True)
        seen = set()
        result: List[int] = []
        for c in ladder:
            if c <= requested_ctx and c not in seen:
                seen.add(c)
                result.append(c)
        # Ensure smallest fallback
        if 8192 not in seen and 8192 <= requested_ctx:
            result.append(8192)
        return result

    @staticmethod
    def _descending_unique(values: Sequence[int]) -> List[int]:
        out: List[int] = []
        for v in values:
            if v > 0 and v not in out:
                out.append(v)
        return sorted(out, reverse=True)


__all__ = ["BaseLlamaCppPipeline", "BaseLlamaCppCore"]

# Alias for non-LangGraph pipelines (embedding, multimodal, etc.)
BaseLlamaCppCore = BaseLlamaCppPipeline
