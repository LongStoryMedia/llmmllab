"""
Base pipeline class with shared timeout protection, circuit breaker, and context management.
"""

import asyncio
import hashlib
import uuid
import logging
from typing import Any, Dict, List, Optional, TypeVar
import math
import re
from collections import deque
from abc import ABC, abstractmethod
from datetime import datetime

from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.language_models import LanguageModelInput, BaseChatModel
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from models import (
    Message,
    MessageRole,
    MessageContent,
    MessageContentType,
    ChatResponse,
    ModelProfile,
    Model,
)
from utils.langgraph import LangGraphState, build_langgraph_state, build_lc_messages
from utils.message import to_lc_message
from utils.langgraph import coerce_to_langchain_message_dict
from runner.pipelines.base import BasePipelineCore, PipeType

T = TypeVar("T")


class CircuitBreakerConfig:
    """Configuration for circuit breaker and timeout protection."""

    def __init__(
        self,
        base_timeout: float = 60.0,  # Base timeout in seconds
        deep_research_timeout: float = 120.0,  # Extended timeout for deep research
        max_retries: int = 2,
        cooldown_period: float = 30.0,  # Time before allowing retry after failure
        # Adaptive generation / perplexity guard settings
        enable_perplexity_guard: bool = True,
        perplexity_window: int = 40,
        perplexity_threshold: float = 10.0,
        avg_logprob_floor: float = -6.0,
        repetition_ngram: int = 6,
        repetition_threshold: int = 6,
        min_tokens_for_eval: Optional[int] = None,
        # Logging / telemetry for adaptive generation
        perplexity_log_interval_tokens: int = 20,
        log_repetition_events: bool = True,
    ):
        self.base_timeout = base_timeout
        self.deep_research_timeout = deep_research_timeout
        self.max_retries = max_retries
        self.cooldown_period = cooldown_period
        # Perplexity guard config
        self.enable_perplexity_guard = enable_perplexity_guard
        self.perplexity_window = perplexity_window
        self.perplexity_threshold = perplexity_threshold
        self.avg_logprob_floor = avg_logprob_floor
        # Allow overrides via environment variables for rapid tuning
        import os as _os  # local import to avoid global dependency at module import

        env_ngram = _os.getenv("REPETITION_NGRAM")
        env_thresh = _os.getenv("REPETITION_THRESHOLD")
        try:
            if env_ngram is not None:
                repetition_ngram = max(2, int(env_ngram))
        except Exception:
            pass
        try:
            if env_thresh is not None:
                repetition_threshold = max(2, int(env_thresh))
        except Exception:
            pass
        self.repetition_ngram = repetition_ngram
        self.repetition_threshold = repetition_threshold
        self.min_tokens_for_eval = (
            min_tokens_for_eval
            if min_tokens_for_eval is not None
            else max(10, perplexity_window // 2)
        )
        self.perplexity_log_interval_tokens = max(5, perplexity_log_interval_tokens)
        self.log_repetition_events = log_repetition_events

        # Tool generation guard settings (more aggressive)
        self.tool_gen_repetition_ngram = 4  # Reduced from 6 for faster detection
        self.tool_gen_repetition_threshold = 3  # Reduced from 4 for faster detection
        env_tool_ngram = _os.getenv("TOOL_GEN_REPETITION_NGRAM")
        env_tool_thresh = _os.getenv("TOOL_GEN_REPETITION_THRESHOLD")
        try:
            if env_tool_ngram is not None:
                self.tool_gen_repetition_ngram = max(5, int(env_tool_ngram))
        except Exception:
            pass
        try:
            if env_tool_thresh is not None:
                self.tool_gen_repetition_threshold = max(3, int(env_tool_thresh))
        except Exception:
            pass


class ModelState:
    """Tracks the state of a model pipeline."""

    def __init__(self):
        self.is_corrupted = False
        self.corruption_time: Optional[datetime] = None
        self.retry_count = 0
        self.last_error: Optional[str] = None


class BaseLangGraphPipeline(BasePipelineCore[PipeType], ABC):
    """
    Base class for LangGraph-based pipelines with built-in timeout protection,
    circuit breaker functionality, and proper resource management.
    """

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        expected_return_type: Optional[type] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
    ):
        super().__init__(model, profile, expected_return_type)
        self.circuit_config = circuit_config or CircuitBreakerConfig()
        self.model_state = ModelState()
        self.memory = MemorySaver()
        self.graph_cache: Dict[int, CompiledStateGraph] = {}
        self._llm_lock = asyncio.Lock()
        self._logger = logging.getLogger(self.__class__.__name__)

        # Abstract attributes that subclasses must implement
        self.llm: Optional[
            Runnable[LanguageModelInput, BaseMessage] | BaseChatModel
        ] = None

    @abstractmethod
    async def _initialize_llm(
        self, gguf_path: str, tools: Optional[List[BaseTool]] = None
    ) -> None:
        """Initialize the LLM. Must be implemented by subclasses."""
        raise NotImplementedError("Subclass must implement _initialize_llm")

    @abstractmethod
    def _get_gguf_path(self) -> str:
        """Get the GGUF file path. Must be implemented by subclasses."""
        raise NotImplementedError("Subclass must implement _get_gguf_path")

    @abstractmethod
    async def _create_system_prompt(
        self, tools: Optional[List[BaseTool]] = None
    ) -> str:
        """Create system prompt with anti-loop instructions. Must be implemented by subclasses."""
        raise NotImplementedError("Subclass must implement _create_system_prompt")

    def _calculate_optimal_gpu_layers(
        self, n_ctx: int, model_size_category: str = "medium"
    ) -> int:
        """
        Dynamically calculate optimal GPU layers based on context size and model category.

        Args:
            n_ctx: Context size in tokens
            model_size_category: Model size category ("small", "medium", "large", "xlarge")
                - "small": ~3B-7B models (e.g., 3B embeddings)
                - "medium": ~13B-20B models (e.g., OpenAI GPT OSS 20B)
                - "large": ~30B-40B models (e.g., Qwen3-Coder 30B)
                - "xlarge": ~70B+ models

        Strategy:
        - Higher context = fewer GPU layers (more KV cache in GPU VRAM)
        - Lower context = more GPU layers (model weights in GPU VRAM)
        - Larger models need more conservative GPU allocation
        """
        # Define base allocations per model size (AGGRESSIVE for research/testing with 48GB VRAM)
        base_allocations = {
            "small": {"max": 99, "high": 95, "medium": 90, "low": 85, "min": 80},
            "medium": {"max": 95, "high": 90, "medium": 85, "low": 80, "min": 75},
            "large": {
                "max": 90,
                "high": 85,
                "medium": 80,
                "low": 75,
                "min": 70,
            },  # Much more aggressive for 30B
            "xlarge": {"max": 85, "high": 80, "medium": 75, "low": 70, "min": 65},
        }

        # Get allocation thresholds for the model category
        alloc = base_allocations.get(model_size_category, base_allocations["medium"])

        # Determine GPU layers based on context size (AGGRESSIVE allocation)
        if n_ctx <= 4096:  # 4K context or less
            return alloc["max"]  # Maximum layers on GPU
        elif n_ctx <= 8192:  # 8K context
            return alloc["high"]  # High allocation
        elif n_ctx <= 16384:  # 16K context
            return alloc["medium"]  # Still aggressive
        elif n_ctx <= 32768:  # 32K context
            return alloc["low"]  # Reasonable allocation
        elif n_ctx <= 65536:  # 64K context
            return alloc["min"]  # Still good allocation
        elif n_ctx <= 131072:  # 128K context (model's trained context)
            return max(
                20, alloc["min"] - 5
            )  # Slightly more conservative but still reasonable
        elif n_ctx <= 262144:  # 256K context
            return max(18, alloc["min"] - 7)  # More conservative for larger context
        elif n_ctx <= 524288:  # 512K context
            return max(15, alloc["min"] - 10)  # Very conservative for large context
        else:  # > 512K context (1M tokens)
            return max(12, alloc["min"] - 13)  # Minimal layers for extreme context

    def _should_use_extended_timeout(self, messages: List[Message]) -> bool:
        """
        Determine if this request should use extended timeout.
        Override in subclasses for custom logic.
        """
        # Check if any message mentions research, web search, or complex analysis
        extended_keywords = [
            "research",
            "web search",
            "analyze",
            "investigate",
            "detailed analysis",
            "comprehensive",
            "deep dive",
            "arduino",
            "code",
            "BOM",
            "bill of materials",
        ]

        for message in messages:
            if message.content:
                for content in message.content:
                    if content.type == MessageContentType.TEXT and content.text:
                        text_lower = content.text.lower()
                        if any(keyword in text_lower for keyword in extended_keywords):
                            return True
        return False

    def _get_timeout_for_request(self, messages: List[Message]) -> float:
        """Get appropriate timeout based on request complexity."""
        if self._should_use_extended_timeout(messages):
            return self.circuit_config.deep_research_timeout
        return self.circuit_config.base_timeout

    async def _check_model_health(self) -> bool:
        """Check if model is in a healthy state for processing."""
        if not self.model_state.is_corrupted:
            return True

        # Check if enough time has passed for cooldown
        if self.model_state.corruption_time:
            time_since_corruption = (
                datetime.now() - self.model_state.corruption_time
            ).total_seconds()
            if time_since_corruption >= self.circuit_config.cooldown_period:
                self._logger.info("Cooldown period elapsed, attempting model reset")
                return await self._reset_model()

        return False

    async def _reset_model(self) -> bool:
        """Reset the model after corruption to restore functionality."""
        try:
            self._logger.info("Resetting corrupted model state...")

            # Clear LLM instance to force reinitialization
            old_llm = self.llm
            self.llm = None

            # Clear graph cache to force rebuild
            self.graph_cache.clear()

            # Try to clean up old LLM resources
            if old_llm is not None:
                try:
                    # Allow some time for cleanup
                    await asyncio.sleep(0.5)
                except Exception as e:
                    self._logger.warning(f"Error during LLM cleanup: {e}")

            # Reset model state
            self.model_state.is_corrupted = False
            self.model_state.corruption_time = None
            self.model_state.retry_count = 0
            self.model_state.last_error = None

            self._logger.info("Model reset completed successfully")
            return True

        except Exception as e:
            self._logger.error(f"Model reset failed: {e}")
            return False

    def _mark_model_corrupted(self, error: str) -> None:
        """Mark the model as corrupted due to timeout or error."""
        self.model_state.is_corrupted = True
        self.model_state.corruption_time = datetime.now()
        self.model_state.retry_count += 1
        self.model_state.last_error = error
        self._logger.error(f"Model marked as corrupted: {error}")

    # ---------------- Adaptive Streaming & Guards -----------------
    def _detect_repetition_smart(
        self, full_response: str, is_tool_generation: bool = False
    ) -> tuple[bool, str]:
        """
        Smart repetition detection that understands context and structure.

        Returns:
            (is_repetitive, reason) - True if repetitive with explanation
        """
        if len(full_response) < 100:  # Too short to analyze
            return False, ""

        # 1. Skip repetition detection in structured content
        if self._is_structured_content(full_response[-500:]):  # Check last 500 chars
            return False, "structured_content_exemption"

        # 2. Check for actual problematic repetition patterns
        if self._detect_token_loops(full_response[-200:]):  # Check last 200 chars
            return True, "token_loop_detected"

        # 3. For tool generation, use stricter semantic checks
        if is_tool_generation:
            if self._detect_json_malformation(full_response[-300:]):
                return True, "malformed_json_repetition"

        # 4. General semantic repetition (less aggressive)
        if len(full_response) > 1000 and self._detect_semantic_repetition_conservative(
            full_response
        ):
            return True, "semantic_repetition"

        return False, ""

    def _is_structured_content(self, text: str) -> bool:
        """Check if the text contains structured content that naturally has repetition."""
        text_lower = text.lower()

        # JSON patterns - use regex to detect any key-value pairs
        if '"' in text and re.search(r'"\w+"\s*:\s*[^,}]+', text):
            return True

        # Table patterns (markdown)
        if "|" in text and any(pattern in text for pattern in ["**", "--", "---"]):
            return True

        # Code blocks
        if any(
            pattern in text
            for pattern in ["```", "`", "def ", "class ", "import ", "function"]
        ):
            return True

        # Lists and structured data
        if text.count("\n") > 3 and any(
            pattern in text for pattern in ["- ", "* ", "1. ", "2. ", "##"]
        ):
            return True

        return False

    def _detect_token_loops(self, text: str) -> bool:
        """Detect actual problematic token loops (same sequence repeating)."""
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) < 10:
            return False

        # Look for exact sequence repetition (more than 4 words repeating 3+ times)
        for seq_len in range(4, min(8, len(words) // 3)):
            for start in range(len(words) - seq_len * 3):
                sequence = words[start : start + seq_len]
                sequence_str = " ".join(sequence)

                # Count how many times this exact sequence appears consecutively
                consecutive_count = 1
                pos = start + seq_len

                while pos + seq_len <= len(words):
                    next_sequence = words[pos : pos + seq_len]
                    if next_sequence == sequence:
                        consecutive_count += 1
                        pos += seq_len
                    else:
                        break

                if consecutive_count >= 3:  # Same sequence 3+ times in a row
                    return True

        return False

    def _detect_json_malformation(self, text: str) -> bool:
        """Detect JSON-specific repetition that indicates malformation."""
        # Try to find if JSON is being generated but malformed
        if '"' not in text:
            return False

        # Count repeated key patterns in JSON-like text using general regex
        json_like_patterns = re.findall(r'"(\w+)"\s*:\s*[^,}]+', text)
        if len(json_like_patterns) > 5:
            # Check if the same key appears too many times
            from collections import Counter

            key_counts = Counter(json_like_patterns)
            for key, count in key_counts.items():
                if count > 3:  # Same key appearing more than 3 times is suspicious
                    return True

        return False

    def _detect_semantic_repetition_conservative(self, text: str) -> bool:
        """Ultra-conservative semantic repetition detection for very obvious cases only."""
        if len(text) < 600:  # Increased from 400 to be more conservative
            return False

        # Only look at the very end of the text to catch active loops
        recent_text = text[-300:]  # Only check last 300 characters

        # Split into sentences and check for near-identical content
        sentences = re.split(r"[.!?]+", recent_text)
        sentences = [
            s.strip() for s in sentences if len(s.strip()) > 30
        ]  # Increased from 20

        if len(sentences) < 3:  # Reduced from 4 - need fewer sentences
            return False

        # Only check for very high similarity (near-exact duplicates)
        recent_sentences = sentences[-3:]  # Only check last 3 sentences
        for i, sent1 in enumerate(recent_sentences):
            for sent2 in recent_sentences[i + 1 :]:
                words1 = set(re.findall(r"\b\w+\b", sent1.lower()))
                words2 = set(re.findall(r"\b\w+\b", sent2.lower()))

                # Both sentences must be substantial and very similar
                if len(words1) > 8 and len(words2) > 8:  # Increased from 5
                    overlap = len(words1 & words2) / min(len(words1), len(words2))
                    if (
                        overlap > 0.95
                    ):  # Increased from 0.85 - only catch near-exact duplicates
                        return True

        return False

    def _detect_semantic_repetition(self, text: str, window_size: int = 50) -> bool:
        """Legacy method - now delegates to smart detection."""
        is_repetitive, _ = self._detect_repetition_smart(text, is_tool_generation=False)
        return is_repetitive

    async def _stream_with_adaptive_controls(
        self, messages: List[Any], is_tool_generation: bool = False
    ) -> AIMessage:
        """Stream generation with repetition + (optional) perplexity monitoring.

        Expects underlying self.llm to support .astream returning chunks whose
        response_metadata may include a "logprobs" dict (OpenAI / llama.cpp w/ logprobs=1).
        Falls back gracefully if logprobs absent.
        """
        if self.llm is None:
            raise RuntimeError("LLM not initialized before streaming")

        cfg = self.circuit_config

        # Log tool generation configuration
        if is_tool_generation:
            self._logger.info(
                f"Tool generation mode enabled. Using smart repetition detection."
            )

        token_logprobs_window: deque[float] = deque(maxlen=cfg.perplexity_window)
        full_response = ""
        tokens_seen = 0
        last_logged_tokens = 0

        # Pre-calculate evaluation thresholds
        min_eval_tokens = cfg.min_tokens_for_eval

        try:
            async for chunk in self.llm.astream(
                messages,
                logprobs=True,  # Request logprobs if supported
                top_logprobs=1,
            ):
                # Content accumulation
                token_text = ""
                if hasattr(chunk, "content") and chunk.content:
                    if isinstance(chunk.content, str):
                        token_text = chunk.content
                    elif isinstance(chunk.content, list):
                        token_text = "".join(str(c) for c in chunk.content)
                    else:
                        token_text = str(chunk.content)
                    full_response += token_text

                # Logprobs collection (if guard enabled)
                if cfg.enable_perplexity_guard:
                    logprobs_data = None
                    # Primary: ChatGenerationChunk.generation_info
                    if hasattr(chunk, "generation_info") and getattr(
                        chunk, "generation_info"
                    ):
                        gi = getattr(chunk, "generation_info") or {}
                        if isinstance(gi, dict):
                            logprobs_data = gi.get("logprobs")
                    # Fallback: some implementations may surface response_metadata
                    if logprobs_data is None and hasattr(chunk, "response_metadata"):
                        meta = getattr(chunk, "response_metadata", None)
                        if isinstance(meta, dict):
                            logprobs_data = meta.get("logprobs")
                    # Extract per-token logprob entries (OpenAI / llama.cpp style)
                    if isinstance(logprobs_data, dict):
                        content_entries = logprobs_data.get("content") or []
                        if isinstance(content_entries, list):
                            for entry in content_entries:
                                if isinstance(entry, dict) and "logprob" in entry:
                                    lp = entry.get("logprob")
                                    if isinstance(lp, (int, float)):
                                        token_logprobs_window.append(float(lp))

                # Smart repetition detection - periodically check for issues
                if (
                    tokens_seen > 50 and tokens_seen % 25 == 0
                ):  # Check every 25 tokens after 50
                    is_repetitive, reason = self._detect_repetition_smart(
                        full_response, is_tool_generation
                    )
                    if is_repetitive:
                        if cfg.log_repetition_events:
                            self._logger.warning(
                                "Smart repetition guard triggered: reason='%s' tokens=%d",
                                reason,
                                tokens_seen,
                            )
                        if reason == "token_loop_detected":
                            full_response += "\n[Paused: repetitive output detected. Please clarify or ask to continue.]"
                        elif reason == "malformed_json_repetition":
                            full_response += "\n[Paused: JSON generation issue detected. Regenerating response.]"
                        else:
                            full_response += "\n[Paused: repetitive content detected. Providing concise response.]"
                        return AIMessage(content=full_response)

                # Periodic perplexity logging (non-guard informational)
                tokens_seen += 1 if token_text else 0
                if (
                    cfg.enable_perplexity_guard
                    and tokens_seen - last_logged_tokens
                    >= cfg.perplexity_log_interval_tokens
                ):
                    avg_logprob_tmp = (
                        sum(token_logprobs_window) / len(token_logprobs_window)
                        if token_logprobs_window
                        else 0.0
                    )
                    rolling_ppl_tmp = math.exp(-avg_logprob_tmp)
                    self._logger.info(
                        "\n\nPerplexity progress: tokens=%d window=%d avg_logprob=%.3f perplexity=%.2f\n\n",
                        tokens_seen,
                        len(token_logprobs_window),
                        avg_logprob_tmp,
                        rolling_ppl_tmp,
                    )
                    last_logged_tokens = tokens_seen

                # Perplexity (rolling) evaluation
                if (
                    cfg.enable_perplexity_guard
                    and len(token_logprobs_window) >= min_eval_tokens
                ):
                    avg_logprob = sum(token_logprobs_window) / len(
                        token_logprobs_window
                    )
                    rolling_perplexity = math.exp(-avg_logprob)
                    if (
                        rolling_perplexity > cfg.perplexity_threshold
                        or avg_logprob < cfg.avg_logprob_floor
                    ):
                        self._logger.info(
                            "Perplexity guard: perplexity=%.2f avg_logprob=%.2f (threshold=%.2f)",
                            rolling_perplexity,
                            avg_logprob,
                            cfg.perplexity_threshold,
                        )
                        full_response += "\n[Generation paused: high model uncertainty detected. Please refine or clarify to continue.]"
                        return AIMessage(content=full_response)

            return AIMessage(content=full_response)
        except Exception as e:  # pragma: no cover
            self._logger.error(f"Adaptive streaming error: {e}")
            # Fallback: attempt non-streaming call if available
            if hasattr(self.llm, "ainvoke"):
                try:
                    return await self.llm.ainvoke(messages)  # type: ignore[attr-defined]
                except Exception as inner:
                    return AIMessage(content=f"Error generating response: {inner}")
            return AIMessage(content=f"Error generating response: {e}")

    async def process_messages(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        is_tool_generation: bool = False,
    ) -> Any:
        """Process messages with circuit breaker protection."""

        # Check model health first
        if not await self._check_model_health():
            error_msg = (
                f"Model is in corrupted state (retry {self.model_state.retry_count}). "
                f"Last error: {self.model_state.last_error or 'Unknown'}. "
                "Please wait for cooldown period or restart the service."
            )
            return self._create_error_response(error_msg)

        # Initialize LLM if needed
        if self.llm is None:
            try:
                gguf_path = self._get_gguf_path()
                await self._initialize_llm(gguf_path, tools)
            except Exception as e:
                error_msg = f"Failed to initialize LLM: {str(e)}"
                self._mark_model_corrupted(error_msg)
                return self._create_error_response(error_msg)

        # Get appropriate timeout for this request
        timeout = self._get_timeout_for_request(messages)
        self._logger.info(f"Using timeout of {timeout}s for this request")

        try:
            # Convert to LangChain messages
            lc_messages = [to_lc_message(msg) for msg in messages]

            # Create and run the graph with timeout protection
            graph = await self._create_graph_with_timeout(
                tools, timeout, is_tool_generation
            )

            # Prepare initial state
            initial_state = build_langgraph_state(
                lc_messages,
                user_input="",
                current_iteration=0,
                max_iterations=10,
                error_count=0,
            )

            # Build runnable config
            thread_source = self._generate_thread_id(messages)
            config = RunnableConfig(
                configurable={"thread_id": f"pipeline-{thread_source}"}
            )

            # Run the graph with timeout protection
            try:
                async with self._llm_lock:
                    result = await asyncio.wait_for(
                        graph.ainvoke(initial_state, config=config), timeout=timeout
                    )
            except asyncio.TimeoutError:
                error_msg = f"Pipeline timed out after {timeout}s - likely infinite loop or complex processing"
                self._logger.error(error_msg)
                self._mark_model_corrupted(error_msg)
                return self._create_error_response(
                    f"Request timed out after {timeout}s. This may be due to complex processing or model issues. "
                    "Please try a simpler request or wait for the system to recover."
                )

            # Extract and return response
            return self._extract_response_from_result(result)

        except Exception as e:
            error_msg = f"Error in process_messages: {str(e)}"
            self._logger.error(error_msg, exc_info=True)

            # Mark as corrupted if it seems like a model issue
            if any(
                keyword in str(e).lower()
                for keyword in ["timeout", "stuck", "loop", "memory", "cuda"]
            ):
                self._mark_model_corrupted(error_msg)

            return self._create_error_response(error_msg)

    async def _create_graph_with_timeout(
        self,
        tools: Optional[List[BaseTool]],
        timeout: float,
        is_tool_generation: bool = False,
    ) -> CompiledStateGraph:
        """Create LangGraph with timeout-aware agent node."""
        tool_signature = hash(tuple(tool.name for tool in (tools or [])))

        if tool_signature in self.graph_cache:
            return self.graph_cache[tool_signature]

        # Ensure LLM initialized
        if self.llm is None:
            gguf_path = self._get_gguf_path()
            await self._initialize_llm(gguf_path, tools)
        else:
            # Bind tools if provided
            if tools:
                try:
                    if not isinstance(self.llm, BaseChatModel):
                        raise TypeError("LLM is not a BaseChatModel")
                    if self.llm.bind_tools:
                        self.llm = self.llm.bind_tools(tools)
                except Exception as e:
                    self._logger.warning(f"Failed to bind tools to LLM: {e}")

        # Build graph with timeout-aware agent node
        workflow = StateGraph(LangGraphState)

        # Create agent node with timeout protection
        async def timeout_protected_agent_node(
            state: LangGraphState, config: RunnableConfig | None = None
        ):
            return await self._agent_node_with_timeout(
                state, config, timeout, is_tool_generation
            )

        workflow.add_node("agent", timeout_protected_agent_node)

        if tools:
            workflow.add_node("tools", ToolNode(tools))
            workflow.add_conditional_edges(
                "agent", tools_condition, {"tools": "tools", END: END}
            )
            workflow.add_edge("tools", "agent")
        else:
            workflow.add_edge("agent", END)

        workflow.add_edge(START, "agent")

        compiled_graph = workflow.compile(checkpointer=self.memory)
        self.graph_cache[tool_signature] = compiled_graph
        return compiled_graph

    async def _agent_node_with_timeout(
        self,
        state: LangGraphState,
        config: RunnableConfig | None,
        timeout: float,
        is_tool_generation: bool = False,
    ) -> Dict[str, Any]:
        """Agent node with built-in timeout protection and repetition detection."""
        # Note: config parameter is reserved for future use by LangGraph
        _ = config  # Silence unused argument warning

        # Check iteration limits
        if state.current_iteration >= state.max_iterations:
            return {
                "messages": [
                    coerce_to_langchain_message_dict(
                        AIMessage(
                            content="I've reached the iteration limit for safety."
                        )
                    )
                ],
                "current_iteration": state.current_iteration + 1,
            }

        if state.error_count >= 3:
            return {
                "messages": [
                    coerce_to_langchain_message_dict(
                        AIMessage(
                            content="I'm experiencing technical difficulties and need to stop."
                        )
                    )
                ],
                "error_count": state.error_count + 1,
            }

        try:
            # Convert messages
            messages = build_lc_messages(state.messages)
            assert self.llm is not None, "LLM not initialized"

            # Use a portion of the total timeout for individual LLM calls
            # Allow pipeline-specific timeouts while preventing infinite loops
            llm_timeout = min(
                timeout * 0.8, max(90.0, timeout)
            )  # Allow up to 80% of pipeline timeout, min 90s

            try:
                # Use the adaptive streaming method which contains the necessary guards
                # and avoids re-entering the pipeline creation lock.
                response = await asyncio.wait_for(
                    self._stream_with_adaptive_controls(messages, is_tool_generation),
                    timeout=llm_timeout,
                )

                # Check for signs of repetitive content in response
                if self._is_response_repetitive(response):
                    self._logger.warning("Detected repetitive response, truncating")
                    if hasattr(response, "content") and isinstance(
                        response.content, str
                    ):
                        # Truncate repetitive response
                        response.content = self._truncate_repetitive_content(
                            response.content
                        )

                return {
                    "messages": [coerce_to_langchain_message_dict(response)],
                    "current_iteration": state.current_iteration + 1,
                }

            except asyncio.TimeoutError as exc:
                self._logger.error(f"LLM invocation timed out after {llm_timeout}s")
                raise RuntimeError(
                    "LLM got stuck in reasoning loop - timeout triggered"
                ) from exc

        except Exception as e:
            error_msg = str(e)
            self._logger.error(f"Agent node error: {error_msg}")

            # Return appropriate error response
            if "timeout" in error_msg.lower() or "stuck in reasoning loop" in error_msg:
                timeout_error = (
                    "The model got stuck in a reasoning loop and was stopped for safety. "
                    "Please try rephrasing your request or starting a new conversation."
                )
                return {
                    "messages": [
                        coerce_to_langchain_message_dict(
                            AIMessage(content=timeout_error)
                        )
                    ],
                    "error_count": state.error_count + 1,
                    "current_iteration": state.current_iteration + 1,
                }
            else:
                return {
                    "messages": [
                        coerce_to_langchain_message_dict(
                            AIMessage(content=f"I encountered an error: {error_msg}")
                        )
                    ],
                    "error_count": state.error_count + 1,
                    "current_iteration": state.current_iteration + 1,
                }

    def _is_response_repetitive(self, response: Any) -> bool:
        """Check if response contains repetitive patterns."""
        if not hasattr(response, "content"):
            return False

        content = response.content
        if not isinstance(content, str) or len(content) < 200:
            return False

        # Check for repeated phrases (simple heuristic)
        words = content.split()
        if len(words) < 50:
            return False

        # Look for repeated sequences of 10+ words
        for i in range(len(words) - 20):
            sequence = " ".join(words[i : i + 10])
            remaining_text = " ".join(words[i + 10 :])
            if sequence in remaining_text:
                self._logger.warning(f"Detected repeated sequence: {sequence[:100]}...")
                return True

        return False

    def _truncate_repetitive_content(self, content: str) -> str:
        """Truncate repetitive content at the first major repetition."""
        # Simple truncation strategy with multiple guards:
        # 1) Limit to first 4 sentences
        # 2) Limit to first 12 lines
        # 3) Remove exact duplicate lines
        text = content
        # Limit lines
        lines = [ln.strip() for ln in text.splitlines()]
        seen = set()
        deduped = []
        for ln in lines:
            if ln and ln not in seen:
                deduped.append(ln)
                seen.add(ln)
        deduped = deduped[:12]
        text = "\n".join(deduped)
        # Limit sentences
        sentences = [
            s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()
        ]
        sentences = sentences[:4]
        truncated = ". ".join(sentences)
        if truncated and not truncated.endswith("."):
            truncated += "."
        return truncated + "\n\n[Response truncated due to repetitive content]"

    def _generate_thread_id(self, messages: List[Message]) -> str:
        """Generate a consistent thread ID for the conversation."""
        try:
            latest_message = messages[-1] if messages else None
            conv_id = getattr(latest_message, "conversation_id", None)
            thread_source = (
                f"{conv_id}-{len(messages)}"
                if conv_id is not None
                else str(uuid.uuid4())
            )
        except Exception:
            thread_source = str(uuid.uuid4())

        return hashlib.md5(thread_source.encode()).hexdigest()[:16]

    def _extract_response_from_result(self, result: Dict[str, Any]) -> Any:
        """Extract response from graph result."""
        if result.get("messages"):
            final_message = result["messages"][-1]
            response_text = ""

            # Support both LangChain AIMessage and dict-shaped messages (from coerce_to_langchain_message_dict)
            if isinstance(final_message, AIMessage):
                content = getattr(final_message, "content", None)
                if content:
                    if isinstance(content, str):
                        response_text = content
                    elif isinstance(content, list):
                        for c in content:
                            if isinstance(c, str):
                                response_text += c + " "
                            elif isinstance(c, dict) and "text" in c:
                                response_text += c["text"] + " "
            elif isinstance(final_message, dict):
                content = final_message.get("content")
                if isinstance(content, str):
                    response_text = content
                elif isinstance(content, list):
                    for c in content:
                        if isinstance(c, str):
                            response_text += c + " "
                        elif isinstance(c, dict) and "text" in c:
                            response_text += c["text"] + " "

            response_text = response_text.strip()

            # Strip hidden chain-of-thought / <think> blocks if model profile disables thinking
            try:
                if not self.profile.parameters.think:
                    # Remove <think>...</think> sections and similar markers
                    response_text = re.sub(
                        r"<think>[\s\S]*?</think>",
                        "",
                        response_text,
                        flags=re.IGNORECASE,
                    )
                    # Also drop leading bracketed meta lines
                    lines = [
                        ln
                        for ln in response_text.splitlines()
                        if not ln.lower().startswith("[think")
                    ]
                    response_text = "\n".join(lines).strip()
            except Exception:
                pass

            # Return appropriate type based on generic parameter
            if self.expected_return_type == str:
                # Guard against pathological repetition in plain text mode
                if len(response_text) > 4000:
                    response_text = self._truncate_repetitive_content(response_text)
                self.validate_return_value(response_text)
                return response_text
            else:
                chat = ChatResponse(
                    done=True,
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT,
                                text=response_text,
                            )
                        ],
                    ),
                    created_at=datetime.now(),
                    finish_reason="stop",
                )
                self.validate_return_value(chat)
                return chat

        # If no response, return error
        return self._create_error_response("No response generated")

    def _create_error_response(self, error_msg: str) -> Any:
        """Create an error response of the appropriate type."""
        if self.expected_return_type == str:
            return error_msg
        else:
            return ChatResponse(
                done=True,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=error_msg,
                        )
                    ],
                ),
                created_at=datetime.now(),
                finish_reason="error",
            )

    def cleanup(self) -> None:
        """Cleanup pipeline resources."""
        if self.llm is not None:
            try:
                # Basic cleanup
                pass
            except Exception as e:
                self._logger.warning(f"Error during cleanup: {e}")
            finally:
                self.llm = None

        # Clear caches
        self.graph_cache.clear()
        self.model_state = ModelState()
