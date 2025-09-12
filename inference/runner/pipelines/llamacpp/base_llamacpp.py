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
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._model_size_category = model_size_category

    # ---------- LLM Initialization (Heuristic Backoff) ----------
    async def _initialize_llm(
        self, gguf_path: str, tools: Optional[List[BaseTool]] = None
    ) -> None:  # noqa: D401
        """Initialize llama.cpp model with heuristic auto-backoff.

        Strategy order (OOM/backoff preference):
            1. Keep context size (n_ctx) constant while reducing n_batch first
            2. Then reduce number of GPU layers (but never below 2/3 of initial allocation)
            3. Only if all batch + gpu layer combinations fail, drop to the next smaller n_ctx
        We do NOT decrement logprobs progressively anymore (removed as ineffective).
        Stops at first successful load.
        """
        if os.environ.get("ALLOW_MISSING_GGUF", "false").lower() in {
            "1",
            "true",
            "yes",
        }:

            class _DummyLLM(BaseChatModel):  # pragma: no cover - dev convenience
                def bind_tools(self, *_, **__):  # type: ignore[override]
                    return self

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

            # Fixed feature settings (always collect logits for all tokens so logprobs works)
            logits_all = True  # user requirement: always True (higher memory usage)
            logprobs = 1  # minimal diagnostics, no progressive reduction

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
                        self.llm = ChatLlamaCpp(
                            model_path=gguf_path,
                            f16_kv=True,
                            n_parts=-1,
                            n_gpu_layers=n_gpu_layers,
                            n_ctx=n_ctx,
                            n_batch=n_batch,
                            use_mmap=True,
                            use_mlock=False,
                            seed=self.profile.parameters.seed or -1,
                            temperature=self.profile.parameters.temperature or 0.7,
                            max_tokens=self.profile.parameters.max_tokens or 4096,
                            top_p=self.profile.parameters.top_p or 0.8,
                            top_k=self.profile.parameters.top_k or 20,
                            repeat_penalty=self.profile.parameters.repeat_penalty
                            or 1.05,
                            stop=self.profile.parameters.stop
                            or ["<|im_end|>", "<|endoftext|>", "<|end|>"],
                            streaming=True,
                            verbose=False,
                            logprobs=logprobs,
                            logits_all=logits_all,
                            callback_manager=CallbackManager(
                                [StreamingStdOutCallbackHandler()]
                            ),
                        )
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
