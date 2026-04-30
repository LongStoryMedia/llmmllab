"""
Reliability nodes for the IDE graph.

Implements a lightweight ML-based reliability stack:
  - BehaviorMonitorNode: Isolation Forest anomaly detection on session trajectory features.
  - ToolVerifierNode:    Logprob + heuristic gate on tool calls before execution.
  - LoopDetectorNode:    LSH / n-gram fingerprint loop detector on agent outputs.

All nodes are non-blocking by default — they annotate WorkflowState and let the
conditional edges decide whether to intervene.  Hard-blocking (raising / short-circuiting
to END) is opt-in via the config flags on each node.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

# sklearn is a soft dependency — nodes degrade gracefully when absent.
try:
    from sklearn.ensemble import IsolationForest

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from graph.state import WorkflowState
from utils.logging import llmmllogger
from utils.message_conversion import extract_text_from_message
from utils.tool_call_types import extract_tool_call_requests


def _tool_call_names(msg: BaseMessage) -> List[str]:
    return [req["name"] for req in extract_tool_call_requests(msg)]


# ---------------------------------------------------------------------------
# BehaviorMonitorNode
# ---------------------------------------------------------------------------


@dataclass
class BehaviorMonitorConfig:
    """Configuration for the behavior monitor."""

    enabled: bool = True
    # Number of messages to use as a feature window.
    window_size: int = 10
    # Isolation Forest contamination — expected fraction of anomalous sessions.
    contamination: float = 0.1
    # Minimum messages before the model is considered fitted.
    min_messages_to_fit: int = 6
    # If True, inject a system warning into state instead of just logging.
    inject_warning: bool = True


def _extract_features(messages: Sequence[BaseMessage], window: int) -> List[float]:
    """
    Compute a fixed-length feature vector from the tail of the message history.

    Features (all normalised to avoid scale dominance):
      0: mean response length (chars) in window — proxy for verbosity drift
      1: tool call rate in window — agentic activity density
      2: unique tool names ratio — diversity of tool usage
      3: ToolMessage error rate — fraction of tool results that are errors
      4: response length variance — instability in output size
    """
    tail = list(messages[-window:]) if len(messages) >= window else list(messages)
    if not tail:
        return [0.0] * 5

    lengths = [len(extract_text_from_message(m)) for m in tail if isinstance(m, AIMessage)] or [0]
    tool_calls = [n for m in tail for n in _tool_call_names(m)]
    tool_msgs = [m for m in tail if isinstance(m, ToolMessage)]
    errors = sum(
        1
        for m in tool_msgs
        if "error" in extract_text_from_message(m).lower() or (getattr(m, "status", None) == "error")
    )

    mean_len = sum(lengths) / len(lengths)
    tool_rate = len(tool_calls) / max(len(tail), 1)
    diversity = len(set(tool_calls)) / max(len(tool_calls), 1)
    error_rate = errors / max(len(tool_msgs), 1)
    variance = (
        sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        if len(lengths) > 1
        else 0.0
    )

    # Soft normalisation — keeps features in a comparable range without hard bounds.
    return [
        math.log1p(mean_len) / 10.0,
        tool_rate,
        diversity,
        error_rate,
        math.log1p(variance) / 15.0,
    ]


class BehaviorMonitorNode:
    """
    Pre-agent anomaly detector.

    Maintains a rolling Isolation Forest fitted on recent session feature vectors.
    On each invocation it:
      1. Extracts features from current message history.
      2. Fits/updates the model when enough samples are available.
      3. Scores the current feature vector.
      4. Annotates state['reliability_flags'] with the anomaly score.

    The node never blocks execution — it only annotates state.
    The conditional edge after it can choose to inject a correction prompt
    or skip straight to the agent.
    """

    NODE_NAME = "behavior_monitor"

    def __init__(self, config: Optional[BehaviorMonitorConfig] = None):
        self.config = config or BehaviorMonitorConfig()
        self.logger = llmmllogger.logger.bind(component="BehaviorMonitorNode")
        self._sample_buffer: Deque[List[float]] = deque(maxlen=200)
        self._model: Optional[Any] = None  # IsolationForest instance

    def _fit_or_update(self, features: List[float]) -> None:
        """Add sample and refit if we have enough data."""
        self._sample_buffer.append(features)
        if not _SKLEARN_AVAILABLE:
            return
        if len(self._sample_buffer) < self.config.min_messages_to_fit:
            return
        self._model = IsolationForest(
            contamination=self.config.contamination,
            random_state=42,
            n_estimators=50,  # Kept small for low latency.
        )
        self._model.fit(list(self._sample_buffer))

    def _score(self, features: List[float]) -> float:
        """
        Return anomaly score in [0, 1] where 1 = most anomalous.
        Returns 0.0 when model is not yet fitted.
        """
        if self._model is None or not _SKLEARN_AVAILABLE:
            return 0.0
        # IsolationForest.score_samples returns negative anomaly scores;
        # more negative = more anomalous.  Normalise to [0, 1].
        raw = self._model.score_samples([features])[0]
        # Typical range is roughly [-0.5, 0.5]; clamp and invert.
        normalised = max(0.0, min(1.0, (-raw + 0.5)))
        return float(normalised)

    async def __call__(self, state: WorkflowState) -> Dict[str, Any]:
        if not self.config.enabled:
            return {}

        messages = state.messages or []
        features = _extract_features(messages, self.config.window_size)
        self._fit_or_update(features)
        score = self._score(features)

        self.logger.debug("Behavior anomaly score", score=score, features=features)

        flags: Dict[str, Any] = state.get("reliability_flags", {}) or {}
        flags["anomaly_score"] = score
        flags["behavior_features"] = features

        updates: Dict[str, Any] = {"reliability_flags": flags}

        if score > 0.7 and self.config.inject_warning:
            # Prepend a soft correction hint into the next agent turn via metadata.
            # The AgentNode should read this and optionally inject a system message.
            flags["behavior_warning"] = (
                "Agent trajectory appears anomalous. "
                "Reconsider the current plan and verify assumptions before proceeding."
            )
            self.logger.warning(
                "High anomaly score — warning injected",
                score=score,
                user_id=state.get("user_id"),
            )

        return updates


# ---------------------------------------------------------------------------
# ToolVerifierNode
# ---------------------------------------------------------------------------


@dataclass
class ToolVerifierConfig:
    """Configuration for the tool verifier."""

    enabled: bool = True
    # Minimum average logprob for a tool call to be trusted.
    # Set to None to skip logprob check (e.g. model doesn't expose them).
    min_avg_logprob: Optional[float] = -3.5
    # Tool names that are always allowed regardless of score.
    always_allow: List[str] = field(default_factory=list)
    # If True, block the tool call and return an error ToolMessage instead.
    hard_block: bool = False
    # Maximum argument string length — unusually long args are suspicious.
    max_arg_length: int = 8192


class ToolVerifierNode:
    """
    Gate node between Agent and ServerToolNode / ToolNode.

    Inspects the last AIMessage's tool_calls and:
      - Checks logprob confidence if available.
      - Validates argument length and basic structure.
      - Annotates state with per-call verdicts.
      - Optionally hard-blocks by replacing tool_calls with an error message.

    Usage in graph:
        workflow.add_node("tool_verifier", ToolVerifierNode())
        # Route: Agent -> tool_verifier -> ServerToolNode -> Agent
    """

    NODE_NAME = "tool_verifier"

    def __init__(self, config: Optional[ToolVerifierConfig] = None):
        self.config = config or ToolVerifierConfig()
        self.logger = llmmllogger.logger.bind(component="ToolVerifierNode")

    def _check_logprob(self, msg: AIMessage) -> Optional[float]:
        """Extract average logprob from response_metadata if present."""
        meta = getattr(msg, "response_metadata", {}) or {}
        # OpenAI-style: logprobs.content[].logprob
        logprobs = meta.get("logprobs", {})
        if logprobs and isinstance(logprobs, dict):
            content = logprobs.get("content", [])
            if content:
                vals = [t.get("logprob", 0.0) for t in content if "logprob" in t]
                return sum(vals) / len(vals) if vals else None
        return None

    def _verdict(
        self, tool_call: Dict[str, Any], avg_logprob: Optional[float]
    ) -> Tuple[bool, str]:
        """
        Return (ok, reason).
        ok=False means the call is suspicious.
        """
        name = tool_call.get("name", "")
        args = tool_call.get("args", {})
        args_str = str(args)

        if name in self.config.always_allow:
            return True, "allowlisted"

        if len(args_str) > self.config.max_arg_length:
            return (
                False,
                f"args too long ({len(args_str)} > {self.config.max_arg_length})",
            )

        if not isinstance(args, dict):
            return False, "args is not a dict"

        if (
            self.config.min_avg_logprob is not None
            and avg_logprob is not None
            and avg_logprob < self.config.min_avg_logprob
        ):
            return False, f"low confidence (avg_logprob={avg_logprob:.3f})"

        return True, "ok"

    async def __call__(self, state: WorkflowState) -> Dict[str, Any]:
        if not self.config.enabled:
            return {}

        messages = state.get("messages", [])
        if not messages:
            return {}

        last = messages[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {}

        avg_logprob = self._check_logprob(last)
        verdicts: List[Dict[str, Any]] = []
        any_blocked = False

        for tc in last.tool_calls:
            ok, reason = self._verdict(tc, avg_logprob)
            verdicts.append({"tool": tc.get("name"), "ok": ok, "reason": reason})
            if not ok:
                any_blocked = True
                self.logger.warning(
                    "Tool call flagged",
                    tool=tc.get("name"),
                    reason=reason,
                    user_id=state.get("user_id"),
                )

        flags: Dict[str, Any] = state.get("reliability_flags", {}) or {}
        flags["tool_verdicts"] = verdicts
        flags["tool_blocked"] = any_blocked

        updates: Dict[str, Any] = {"reliability_flags": flags}

        if any_blocked and self.config.hard_block:
            # Inject a synthetic ToolMessage error so the agent self-corrects
            # without executing the suspect call.
            blocked_names = [v["tool"] for v in verdicts if not v["ok"]]
            error_msg = ToolMessage(
                content=(
                    f"Tool call(s) {blocked_names} were blocked by the reliability "
                    f"verifier. Reasons: "
                    + "; ".join(
                        f"{v['tool']}: {v['reason']}" for v in verdicts if not v["ok"]
                    )
                    + ". Please reconsider the approach."
                ),
                tool_call_id="reliability-verifier",
            )
            updates["messages"] = messages + [error_msg]

        return updates


# ---------------------------------------------------------------------------
# LoopDetectorNode
# ---------------------------------------------------------------------------


@dataclass
class LoopDetectorConfig:
    """Configuration for the loop detector."""

    enabled: bool = True
    # n-gram size for structural fingerprinting of tool call sequences.
    ngram_size: int = 4
    # How many recent tool-call sequence n-grams to keep.
    history_size: int = 50
    # Repetition count at which a loop is declared.
    repetition_threshold: int = 3
    # Semantic similarity threshold for output embedding loop detection.
    # Set to None to skip embedding check.
    embedding_similarity_threshold: Optional[float] = 0.95
    # If True, inject a loop-break prompt when a loop is detected.
    inject_break_prompt: bool = True
    # If True, return END signal (hard-stop the loop).
    hard_stop: bool = False


def _ngrams(seq: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]


def _simhash(text: str, bits: int = 64) -> int:
    """
    Simple SimHash for near-duplicate detection without external deps.
    Groups of 4-character shingles → bit vector → hash.
    """
    v = [0] * bits
    shingles = [text[i : i + 4] for i in range(len(text) - 3)] or [text]
    for shingle in shingles:
        h = int(hashlib.md5(shingle.encode()).hexdigest(), 16)
        for i in range(bits):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1
    return sum(1 << i for i in range(bits) if v[i] > 0)


def _hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class LoopDetectorNode:
    """
    Post-agent loop detector.

    Maintains two parallel detectors:
      1. Structural: n-gram fingerprints on the sequence of tool call names.
         Catches "plan → read → plan → read" cycles even with different arguments.
      2. Semantic: SimHash on raw agent text output.
         Catches paraphrase loops where the agent rewrites the same failed response.

    On detection, annotates state['reliability_flags']['loop_detected'] = True
    and optionally injects a loop-break instruction.
    """

    NODE_NAME = "loop_detector"

    def __init__(self, config: Optional[LoopDetectorConfig] = None):
        self.config = config or LoopDetectorConfig()
        self.logger = llmmllogger.logger.bind(component="LoopDetectorNode")
        self._ngram_counts: Counter = Counter()
        self._simhash_history: Deque[int] = deque(maxlen=20)

    def _check_structural_loop(self, messages: Sequence[BaseMessage]) -> bool:
        """Check tool call sequence for repeated n-grams."""
        tool_seq = [name for m in messages for name in _tool_call_names(m)]
        if len(tool_seq) < self.config.ngram_size:
            return False
        new_grams = _ngrams(tool_seq, self.config.ngram_size)
        for gram in new_grams:
            self._ngram_counts[gram] += 1
            if self._ngram_counts[gram] >= self.config.repetition_threshold:
                self.logger.warning(
                    "Structural loop detected",
                    gram=gram,
                    count=self._ngram_counts[gram],
                )
                return True
        # Trim old counts to prevent memory growth
        if len(self._ngram_counts) > self.config.history_size:
            least = self._ngram_counts.most_common()[-(len(self._ngram_counts) // 2) :]
            for k, _ in least:
                del self._ngram_counts[k]
        return False

    def _check_semantic_loop(self, messages: Sequence[BaseMessage]) -> bool:
        """Check last agent output for near-duplicate similarity to recent outputs."""
        ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
        if not ai_msgs:
            return False
        last_text = extract_text_from_message(ai_msgs[-1])
        if not last_text.strip():
            return False
        h = _simhash(last_text)
        for prev_h in self._simhash_history:
            # 64-bit simhash: distance < 4 is a near-duplicate.
            if _hamming_distance(h, prev_h) < 4:
                self.logger.warning("Semantic loop detected via SimHash")
                return True
        self._simhash_history.append(h)
        return False

    async def __call__(self, state: WorkflowState) -> Dict[str, Any]:
        if not self.config.enabled:
            return {}

        messages = state.get("messages", [])
        structural = self._check_structural_loop(messages)
        semantic = self._check_semantic_loop(messages)
        loop_detected = structural or semantic

        flags: Dict[str, Any] = state.get("reliability_flags", {}) or {}
        flags["loop_detected"] = loop_detected
        flags["structural_loop"] = structural
        flags["semantic_loop"] = semantic

        updates: Dict[str, Any] = {"reliability_flags": flags}

        if loop_detected and self.config.inject_break_prompt:
            kind = "structural" if structural else "semantic"
            self.logger.warning(
                "Loop detected — injecting break prompt",
                kind=kind,
                user_id=state.get("user_id"),
            )
            flags["loop_break_prompt"] = (
                f"A {kind} loop has been detected in your recent actions. "
                "Stop repeating the current strategy. Reassess the goal, "
                "try a fundamentally different approach, or ask for clarification."
            )

        return updates


# ---------------------------------------------------------------------------
# Conditional edge helpers
# ---------------------------------------------------------------------------


def should_skip_after_monitor(state: WorkflowState) -> str:
    """
    Conditional edge: BehaviorMonitor -> agent | intervention.
    Returns 'agent' normally; 'intervene' when anomaly is very high.
    Extend this to add a correction/reflection node if desired.
    """
    flags = state.get("reliability_flags") or {}
    score = flags.get("anomaly_score", 0.0)
    return "intervene" if score > 0.85 else "agent"


def should_stop_after_loop_check(state: WorkflowState) -> str:
    """
    Conditional edge: LoopDetector -> end | agent.
    When hard_stop is desired, route to END on loop detection.
    """
    flags = state.get("reliability_flags") or {}
    return "end" if flags.get("loop_detected") else "agent"
