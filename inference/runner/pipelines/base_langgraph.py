"""
Base pipeline class with shared timeout protection, circuit breaker, and context management.
"""

import os
import re
import logging
import math
from typing import Any, Dict, List, Optional, TypeVar, AsyncIterator
import asyncio
from collections import deque
from abc import ABC, abstractmethod
from datetime import datetime

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
    CircuitBreakerConfig,
)
from models.default_configs import DEFAULT_CIRCUIT_BREAKER_CONFIG
from utils.langgraph import (
    LangGraphState,
    build_langgraph_state,
    build_lc_messages,
    coerce_to_langchain_message_dict,
)
from utils.message import to_lc_message
from runner.pipelines.base import BasePipelineCore, PipeType

T = TypeVar("T")


def apply_circuit_breaker_env_overrides(
    config: CircuitBreakerConfig,
) -> CircuitBreakerConfig:
    """Apply environment variable overrides to CircuitBreakerConfig."""
    import os

    # Create a dict with current config values
    config_dict = config.model_dump()

    # Apply environment variable overrides
    env_enable = os.getenv("ENABLE_REPETITION_DETECTION")
    env_ngram = os.getenv("REPETITION_NGRAM")
    env_thresh = os.getenv("REPETITION_THRESHOLD")
    env_tool_ngram = os.getenv("TOOL_GEN_REPETITION_NGRAM")
    env_tool_thresh = os.getenv("TOOL_GEN_REPETITION_THRESHOLD")

    try:
        if env_enable is not None:
            config_dict["enable_repetition_detection"] = env_enable.lower() in (
                "true",
                "1",
                "yes",
                "on",
            )
    except Exception:
        pass

    try:
        if env_ngram is not None:
            config_dict["repetition_ngram"] = max(2, int(env_ngram))
    except Exception:
        pass

    try:
        if env_thresh is not None:
            config_dict["repetition_threshold"] = max(2, int(env_thresh))
    except Exception:
        pass

    try:
        if env_tool_ngram is not None:
            config_dict["tool_gen_repetition_ngram"] = max(2, int(env_tool_ngram))
    except Exception:
        pass

    try:
        if env_tool_thresh is not None:
            config_dict["tool_gen_repetition_threshold"] = max(2, int(env_tool_thresh))
    except Exception:
        pass

    # Return new config instance with overrides applied
    return CircuitBreakerConfig(**config_dict)


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

        # Use the provided circuit config or fallback to default
        # Server should handle all merging logic (defaults → global → user → profile)
        base_circuit_config = circuit_config or DEFAULT_CIRCUIT_BREAKER_CONFIG
        self.circuit_config = apply_circuit_breaker_env_overrides(base_circuit_config)
        self.model_state = ModelState()
        self.memory = MemorySaver()
        self.graph_cache: Dict[int, CompiledStateGraph] = {}
        self._llm_lock = asyncio.Lock()
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.DEBUG)

        # Debug logging for pipeline instantiation
        self._logger.debug(
            f"Pipeline instantiated: model={model.model}, \n"
            f"model: model={model.model_dump_json()}, \n"
            f"profile: profile={profile.model_dump_json()}, \n"
            f"model_id={getattr(model, 'id', 'unknown')}, \n"
            f"model_name={getattr(model, 'name', 'unknown')}, \n"
            f"provider={getattr(model, 'provider', 'unknown')}, \n"
            f"task={getattr(model, 'task', 'unknown')}, \n"
            f"profile_model_name={profile.model_name}, \n"
            f"expected_return_type={expected_return_type}, \n"
            f"circuit_config={circuit_config}, \n"
            f"profile_parameters={profile.parameters.model_dump_json()}, \n"
            f"profile_system_prompt_length={len(profile.system_prompt) if profile.system_prompt else 0}\n"
        )

        # Abstract attributes that subclasses must implement
        self.llm: Optional[
            Runnable[LanguageModelInput, BaseMessage] | BaseChatModel
        ] = None

    @abstractmethod
    async def _initialize_llm(
        self, gguf_path: str, tools: Optional[List[BaseTool]] = None, grammar: Optional[str] = None
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

    # ---- Tool Calling Helper Methods ----
    def _create_tool_description(self, tools: Optional[List[BaseTool]]) -> str:
        """Create standardized tool descriptions for system prompts."""
        if not tools:
            return ""

        tool_descriptions = []
        for tool in tools:
            tool_descriptions.append(f"- {tool.name}: {tool.description}")

        return f"""

Available tools:
{chr(10).join(tool_descriptions)}

Use these tools when they can help provide more accurate or comprehensive responses."""

    def _create_standard_agent_node(self, use_harmony_format: bool = False):
        """Create a standard agent node following LangGraph best practices.

        This can be used by subclasses to implement consistent agent behavior
        with proper tool calling integration.

        Args:
            use_harmony_format: If True, enables special formatting for harmony-compatible models
        """

        async def agent_node(state: LangGraphState, config=None) -> Dict[str, Any]:
            _ = config

            # Iteration guard
            if state.current_iteration >= state.max_iterations:
                msg = f"Maximum iterations ({state.max_iterations}) reached. Stopping to prevent infinite loops."
                return {
                    "messages": [
                        coerce_to_langchain_message_dict(AIMessage(content=msg))
                    ],
                    "current_iteration": state.current_iteration + 1,
                }

            try:
                # Initialize LLM if not done yet
                if self.llm is None:
                    gguf_path = self._get_gguf_path()
                    await self._initialize_llm(gguf_path)

                # Build messages for LLM using standard LangChain format
                messages = build_lc_messages(state.messages)

                # Use base class streaming with timeout and safety controls
                response = await self._stream_with_adaptive_controls(
                    messages, is_tool_generation=False
                )

                return {
                    "messages": [coerce_to_langchain_message_dict(response)],
                    "current_iteration": state.current_iteration + 1,
                }

            except Exception as e:
                error_msg = f"Error in agent node: {str(e)}"
                self._logger.error(error_msg, exc_info=True)
                return {
                    "messages": [
                        coerce_to_langchain_message_dict(AIMessage(content=error_msg))
                    ],
                    "current_iteration": state.current_iteration + 1,
                }

        return agent_node

    # ---- Channel Extraction Methods ----
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
            return self.circuit_config.deep_research_timeout or 120.0
        return self.circuit_config.base_timeout or 60.0

    async def _check_model_health(self) -> bool:
        """Check if model is in a healthy state for processing."""
        if not self.model_state.is_corrupted:
            return True

        # Check if enough time has passed for cooldown
        if self.model_state.corruption_time:
            time_since_corruption = (
                datetime.now() - self.model_state.corruption_time
            ).total_seconds()
            cooldown_period = self.circuit_config.cooldown_period or 30.0
            if time_since_corruption >= cooldown_period:
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
        # For code generation, check the entire response for structured patterns
        if is_tool_generation:
            # When generating tools/code, be more lenient about structured content
            if self._is_code_generation(full_response):
                return False, "code_generation_exemption"

        # Check broader context for structured content, not just the end
        check_length = min(800, len(full_response))  # Check up to 800 chars
        if self._is_structured_content(
            full_response[-check_length:]
        ):  # Check last portion
            return False, "structured_content_exemption"

        # Also check the beginning for dictionary assignments or similar patterns
        if len(full_response) > 200:
            beginning_check = full_response[:200]
            if re.search(r"[\w_]+\s*=\s*\{", beginning_check):
                return False, "structured_content_at_start"

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

    def _is_code_generation(self, text: str) -> bool:
        """
        Check if we're generating code content.
        Look for patterns that suggest code generation.
        """
        # Look for common code patterns
        code_patterns = [
            r"def\s+\w+\s*\(",  # Function definitions
            r"class\s+\w+\s*[\(:]",  # Class definitions
            r"import\s+\w+",  # Import statements
            r"from\s+\w+\s+import",  # From imports
            r"[\w_]+\s*=\s*\{",  # Dictionary assignments
            r"[\w_]+\s*=\s*\[",  # List assignments
            r"```\w*\n",  # Code blocks
            r"if\s+__name__\s*==",  # Main check
            r"try:\s*\n",  # Try blocks
            r"except\s+\w+",  # Exception handling
            r"return\s+[\w\{\[]",  # Return statements with objects
        ]

        # Check if any code pattern is present
        for pattern in code_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True

        # Also check for multiple dictionary/function patterns which is common in code gen
        dict_count = len(re.findall(r"[\w_]+\s*=\s*\{", text))
        if dict_count >= 2:  # Multiple dictionary assignments suggest code generation
            return True

        return False

    def _is_structured_content(self, text: str) -> bool:
        """Check if the text contains structured content that naturally has repetition."""

        # JSON patterns - use regex to detect any key-value pairs
        if '"' in text and re.search(r'"\w+"\s*:\s*[^,}]+', text):
            return True

        # Python dictionary patterns (both quoted and unquoted keys)
        if re.search(r"\w+\s*:\s*[^,}]+", text) and any(
            char in text for char in ["{", "}", ":"]
        ):
            return True

        # Dictionary assignment patterns - be more aggressive about detecting these
        # This catches patterns like "party_plan = {", "config = {", etc.
        if re.search(r"[\w_]+\s*=\s*\{", text):
            return True

        # Multiple dictionary assignments in sequence (very common in code generation)
        dict_assignments = re.findall(r"[\w_]+\s*=\s*\{", text)
        if len(dict_assignments) >= 2:
            return True

        # Class/function definitions with dictionaries
        if re.search(r"(def|class)\s+\w+.*:\s*\{", text, re.MULTILINE):
            return True

        # Multi-line dictionary patterns (common in generated code)
        if (
            "{" in text
            and text.count("\n") > 2
            and re.search(r'["\']?\w+["\']?\s*:\s*[^,}]+', text)
        ):
            return True

        # Python class method patterns with similar structure
        if re.search(r"def\s+\w+\(self.*?\):", text) and "{" in text:
            return True

        # Function/method calls with dictionaries as parameters
        if re.search(r"\w+\([^)]*\{[^}]*\}[^)]*\)", text):
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
                # Check if this sequence repeats at least 3 times

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
            for count in key_counts.values():
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

    def _detect_semantic_repetition(self, text: str, _window_size: int = 50) -> bool:
        """Legacy method - now delegates to smart detection."""
        is_repetitive, _ = self._detect_repetition_smart(text, is_tool_generation=False)
        return is_repetitive

    def _detect_emergency_repetition(self, text: str) -> bool:
        """
        Emergency detection for extreme repetition patterns that should always be caught.
        This is a very conservative check for only the most obvious repetition cases.
        """
        if len(text) < 1000:  # Increased from 500 to 1000 - need more text to be sure
            return False

        # Debug logging to help troubleshoot false positives
        self._logger.debug(f"Emergency repetition check on {len(text)} chars")

        # Check last 800 characters for extreme repetition (increased from 500)
        check_text = text[-800:].lower()

        # Look for very simple repetitive patterns
        words = check_text.split()
        if len(words) < 30:  # Increased from 15 to 30 - need more words to be confident
            return False

        # Only check for extreme word repetition (much higher threshold)
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Increased from 2 to 3 - skip short common words
                word_counts[word] = word_counts.get(word, 0) + 1

        # If any single word appears more than 20 times in last 800 chars, it's extreme repetition
        # Increased from 12 to 20 times - much more conservative
        max_count = max(word_counts.values()) if word_counts else 0
        if max_count > 20:
            # Find which word is repeating
            repeat_word = max(word_counts, key=lambda word: word_counts[word])
            self._logger.debug(
                f"Emergency repetition: word '{repeat_word}' appears {max_count} times"
            )
            return True

        # Check for extremely repeated phrases (be much more conservative)
        text_parts = check_text.replace("\n", " ").split()
        for i in range(len(text_parts) - 6):  # Increased minimum phrase length
            phrase = " ".join(text_parts[i : i + 3])  # 3-word phrases instead of 2
            if len(phrase) > 10:  # Increased minimum phrase length from 5 to 10
                # Count occurrences of this phrase in the remaining text
                remaining_text = " ".join(text_parts[i:])
                count = remaining_text.count(phrase)
                if count > 15:  # Increased from 8 to 15 - same 3-word phrase 15+ times
                    self._logger.debug(
                        f"Emergency repetition: phrase '{phrase}' appears {count} times"
                    )
                    return True

        # Check for extremely obvious patterns like "the the the the the"
        consecutive_repeats = 0
        for i in range(len(words) - 1):
            if words[i] == words[i + 1] and len(words[i]) > 2:
                consecutive_repeats += 1
                if consecutive_repeats > 8:  # 8+ consecutive identical words
                    self._logger.debug(
                        f"Emergency repetition: consecutive word '{words[i]}' repeated {consecutive_repeats + 1} times"
                    )
                    return True
            else:
                consecutive_repeats = 0

        return False

    async def _stream_with_adaptive_controls(
        self, messages: List[Any], is_tool_generation: bool = False
    ) -> AIMessage:
        """
        Stream response with adaptive controls, circuit breaker, and repetition detection.
        Returns an AIMessage with the complete response content.
        """
        if self.llm is None:
            raise RuntimeError("LLM not initialized before streaming")

        cfg = self.circuit_config

        # Log tool generation configuration
        if is_tool_generation:
            self._logger.info(
                "Tool generation mode enabled. Using smart repetition detection."
            )

        token_logprobs_window: deque[float] = deque(maxlen=cfg.perplexity_window)
        full_response = ""
        tokens_seen = 0
        last_logged_tokens = 0

        try:
            # Only request logprobs if perplexity guard is enabled and model supports it
            stream_kwargs = {}
            if cfg.enable_perplexity_guard:
                stream_kwargs["logprobs"] = True
                stream_kwargs["top_logprobs"] = 1

            async for chunk in self.llm.astream(messages, **stream_kwargs):
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
                    cfg.enable_repetition_detection
                    and tokens_seen > 50
                    and tokens_seen % 25 == 0
                ):
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

                # Emergency repetition detection
                elif tokens_seen > 500 and tokens_seen % 200 == 0:
                    if self._detect_emergency_repetition(full_response):
                        self._logger.warning(
                            "Emergency repetition guard triggered at %d tokens. Last 200 chars: %s",
                            tokens_seen,
                            repr(full_response[-200:]),
                        )
                        full_response += "\n[Error: Extreme repetitive output detected. Stopping generation.]"
                        return AIMessage(content=full_response)

                # Periodic perplexity logging
                tokens_seen += 1 if token_text else 0
                log_interval = cfg.perplexity_log_interval_tokens or 20
                if (
                    cfg.enable_perplexity_guard
                    and tokens_seen - last_logged_tokens >= log_interval
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

                # Perplexity evaluation
                min_eval_tokens = cfg.min_tokens_for_eval or 20
                if (
                    cfg.enable_perplexity_guard
                    and len(token_logprobs_window) >= min_eval_tokens
                ):
                    avg_logprob = sum(token_logprobs_window) / len(
                        token_logprobs_window
                    )
                    rolling_perplexity = math.exp(-avg_logprob)
                    perplexity_threshold = cfg.perplexity_threshold or 10.0
                    logprob_floor = cfg.avg_logprob_floor or -6.0
                    if (
                        rolling_perplexity > perplexity_threshold
                        or avg_logprob < logprob_floor
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
        except Exception as e:
            self._logger.error(f"Adaptive streaming error: {e}")
            # Fallback: attempt non-streaming call if available
            if hasattr(self.llm, "ainvoke"):
                try:
                    result = await self.llm.ainvoke(messages)
                    if isinstance(result, AIMessage):
                        return result
                    else:
                        return AIMessage(content=str(result))
                except Exception as inner:
                    return AIMessage(content=f"Error generating response: {inner}")
            return AIMessage(content=f"Error generating response: {e}")

    async def _create_graph_with_timeout(
        self,
        tools: Optional[List[BaseTool]],
        timeout: float,
        is_tool_generation: bool = False,
        grammar: Optional[str] = None,
    ) -> CompiledStateGraph:
        """Create LangGraph with timeout-aware agent node."""
        tool_signature = hash(tuple(tool.name for tool in (tools or [])))

        if tool_signature in self.graph_cache:
            return self.graph_cache[tool_signature]

        # Ensure LLM initialized
        if self.llm is None:
            gguf_path = self._get_gguf_path()
            await self._initialize_llm(gguf_path, tools, grammar)
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
                del self.llm
            except Exception as e:
                self._logger.warning(f"Error during cleanup: {e}")
            finally:
                self.llm = None

        # Clear caches
        self.graph_cache.clear()
        self.model_state = ModelState()
