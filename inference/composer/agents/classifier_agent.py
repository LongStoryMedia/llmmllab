"""
Intent analysis and classification agent.
Performs comprehensive intent analysis following the capability-driven architecture.
Maps user requests to RequiredCapabilities and assesses computational complexity.
"""

import json
from typing import List, TYPE_CHECKING, Optional

from pydantic import BaseModel

from models import (
    CircuitBreakerConfig,
    IntentAnalysis,
    ComplexityLevel,
    ModelProfile,
    PipelinePriority,
    Message,
    MessageRole,
)
from composer.monitoring.logging import composer_logger
from composer.core.errors import IntentAnalysisError
from utils.message import extract_message_text
from utils.grammar_generator import parse_structured_output

if TYPE_CHECKING:
    from runner import PipelineFactory


class ClassifierAgent:
    """
    Grammar-constrained LLM intent analysis agent for workflow routing and tool selection.

    This agent performs comprehensive intent analysis using structured LLM output with
    grammar constraints to ensure guaranteed schema validation. It analyzes user messages
    to determine primary intent, complexity levels, required capabilities, and computational
    requirements that drive LangGraph workflow routing decisions.

    Key Features:
    - Grammar-constrained structured output using llamacpp grammars
    - Type-safe IntentAnalysis model generation with schema validation
    - Adaptive search depth determination based on complexity assessment
    - Graceful fallback handling when model profiles unavailable
    - Integration with shared model profile utilities for configuration management

    The agent executes early in LangGraph workflows to guide tool selection,
    workflow type selection, and retrieval depth configuration through structured
    intent classification that eliminates JSON parsing errors.
    """

    def __init__(
        self,
        pipeline_factory: "PipelineFactory",
        profile: ModelProfile,
    ):
        """
        Initialize the intent classification agent.

        """
        self.pipeline_factory = pipeline_factory
        self.profile = profile
        composer_logger.logger.info(
            "Intent classifier initialized with analysis model profile"
        )

    def determine_search_depth(self, intent_analysis: IntentAnalysis) -> str:
        """
        Determine search depth configuration for adaptive retrieval routing.

        Maps IntentAnalysis complexity levels to search execution paths:
        - COMPLEX/SPECIALIZED → "deep" (multi-step crawl and synthesis)
        - MODERATE → "moderate" (enhanced retrieval with limited sources)
        - TRIVIAL/SIMPLE → "shallow" (single-pass vector store retrieval)

        This drives LangGraph conditional routing between:
        - execute_deep_crawl_and_synthesize node
        - execute_shallow_search node

        Args:
            intent_analysis: Structured intent analysis from grammar-constrained LLM

        Returns:
            str: Search depth configuration ("shallow", "moderate", "deep")
        """
        if intent_analysis.complexity_level in [
            ComplexityLevel.COMPLEX,
            ComplexityLevel.SPECIALIZED,
        ]:
            return "deep"
        elif intent_analysis.complexity_level == ComplexityLevel.MODERATE:
            return "moderate"
        else:
            return "shallow"

    async def analyze(
        self,
        current_user_message: Message,
        circuit_breaker: Optional[CircuitBreakerConfig] = None,
    ) -> List[IntentAnalysis]:
        """
        Perform grammar-constrained intent analysis using structured LLM output.

        Extracts the latest user message from conversation history and analyzes it using
        a specialized analysis model profile. The LLM output is grammar-constrained to
        ensure guaranteed structure validation matching the IntentAnalysis schema.

        The analysis produces structured intent classification including primary intent,
        complexity assessment, required capabilities, and computational requirements.
        This structured output drives LangGraph conditional routing for workflow selection.

        Args:
            user_id: User ID for model profile retrieval
            messages: Conversation messages list

        Returns:
            IntentAnalysis: Validated structured analysis for workflow routing

        Raises:
            IntentAnalysisError: When analysis fails or user message unavailable
        """
        try:

            # Extract user message text
            user_query = extract_message_text(current_user_message)

            # Use pipeline with default priority for intent analysis
            with self.pipeline_factory.pipeline(
                self.profile, str, PipelinePriority.HIGH, circuit_breaker
            ) as pipeline:
                intent_analyses = await self._llm_analyze_intent(pipeline, user_query)
                # Apply any statistical augmentations
                for intent_analysis in intent_analyses:
                    self._augment_with_statistics(intent_analysis, user_query)

                    composer_logger.log_intent_analysis(
                        intent_result={
                            "primary_intent": intent_analysis.primary_intent,
                            "complexity": intent_analysis.complexity_level,
                            "capabilities_count": len(
                                intent_analysis.required_capabilities
                            ),
                        },
                        confidence=intent_analysis.confidence,
                    )

                return intent_analyses

        except Exception as e:
            composer_logger.log_error(e, {"context": "intent_analysis"})
            raise IntentAnalysisError(f"Intent analysis failed: {e}") from e

    async def _llm_analyze_intent(
        self, pipeline, user_query: str
    ) -> List[IntentAnalysis]:
        """
        Execute grammar-constrained LLM analysis with structured output.

        intent_analysis.yaml schema to ensure guaranteed structure validation.
        Current implementation uses JSON schema validation as fallback.

        Args:
            pipeline: Model pipeline with analysis profile
            user_query: User message text for analysis

        Returns:
            str: LLM response (should be grammar-constrained JSON)
        """

        class _Intnts(BaseModel):
            intents: List[IntentAnalysis]

        analysis_prompt = f"""
You are an expert intent classification system. Analyze the user request and output ONLY JSON.

Enumerations (must use exactly these values where applicable):
    primary_intent: general | research | engineering | creative | image_generation | image_refinement
    complexity_level: TRIVIAL | SIMPLE | MODERATE | COMPLEX | SPECIALIZED
    required_capabilities (array, choose relevant): basic_math, text_processing, information_retrieval, conversation_memory, web_search, summarization, reasoning, general_knowledge, api_integration, async_processing, file_manipulation, data_processing, image_processing, audio_processing, database_access, network_communication
    computational_requirements (array, choose relevant): high_memory, gpu_acceleration, parallel_processing, real_time_processing, large_data_handling, complex_reasoning, multi_modal_processing, external_api_calls, file_operations, database_operations

Instructions:
1. Decompose only if there are clearly separable sub-tasks; else one intent.
2. Each element in intents must follow the enumerations exactly.
3. domain_specificity, reusability_potential, confidence are floats 0.0-1.0.
4. Omit response_format / technical_domain unless clearly implied.
5. Output strictly valid JSON. No prose, no markdown, no comments.

User Request: {user_query}

Return JSON that is valid against this schema:
{json.dumps(_Intnts.model_json_schema())}

If multiple intents are needed, include additional objects in the intents array.
"""
        from runner import run_pipeline  # pylint: disable=import-outside-toplevel

        try:
            result = await run_pipeline(
                messages=analysis_prompt,
                pipeline=pipeline,
                tools=None,
                grammar=_Intnts,
            )
        except Exception as e:  # pragma: no cover - pipeline invocation failure
            # Treat failure as hard error (no silent fallback) so tests catch it
            raise IntentAnalysisError(
                f"Intent analysis model invocation failed: {e}"
            ) from e

        txt = extract_message_text(result.message) if result and result.message else ""
        if not txt.strip():
            raise IntentAnalysisError("Empty intent analysis response")

        # Attempt structured parse with minimal JSON repair before giving up
        try:
            intents = parse_structured_output(txt, _Intnts)
            return intents.intents
        except Exception as e:
            # Attempt lightweight repair: truncate to last complete object and close array/object
            repaired = self._attempt_json_repair(txt)
            if repaired and repaired != txt:
                try:
                    intents = parse_structured_output(repaired, _Intnts)
                    composer_logger.logger.info(
                        "Intent JSON repaired successfully",
                        extra={"original_len": len(txt), "repaired_len": len(repaired)},
                    )
                    return intents.intents
                except Exception as e2:  # still failing
                    raise IntentAnalysisError(
                        f"Intent parsing failed after repair: {e2} (original error: {e})"
                    ) from e2
            # No repair success
            raise IntentAnalysisError(
                f"Intent parsing failed: {e}. Raw (truncated): {txt[:200]}"
            ) from e

    def _augment_with_statistics(
        self, intent_analysis: IntentAnalysis, user_query: str
    ):
        """Augment LLM analysis with statistical insights where beneficial."""
        # Apply query length adjustments to confidence
        query_length = len(user_query)
        confidence_adjustment = 0.0

        if query_length < 10:
            confidence_adjustment -= 0.1  # Short queries are harder to classify
        elif query_length > 200:
            confidence_adjustment -= 0.05  # Very long queries may be ambiguous
        else:
            confidence_adjustment += 0.05  # Good length for analysis

        # Adjust confidence based on capability count consistency
        num_capabilities = len(intent_analysis.required_capabilities)
        if num_capabilities == 0:
            confidence_adjustment -= 0.2  # No capabilities is suspicious
        elif num_capabilities > 6:
            confidence_adjustment -= (
                0.1  # Too many capabilities may indicate over-classification
            )

        # Create updated analysis with adjusted confidence
        adjusted_confidence = max(
            0.1, min(1.0, intent_analysis.confidence + confidence_adjustment)
        )

        intent_analysis.confidence = adjusted_confidence

    def _attempt_json_repair(self, raw: str) -> str:
        """Attempt minimal JSON repair for truncated intents array.

        Strategy: Find the last complete object terminator '}' before any dangling comma,
        ensure the prefix up to that brace is kept, then close the intents array and root object.
        Only applied if raw starts with '{' and contains '"intents"'.
        """
        import re

        if not raw.startswith("{") or '"intents"' not in raw:
            return raw
        # Remove any trailing non-JSON characters
        trimmed = raw.strip()
        # If already ends properly, return as-is
        if trimmed.endswith("}"):  # might already be valid
            return trimmed
        # Find last complete object '}'
        last_obj = trimmed.rfind("}")
        if last_obj == -1:
            return raw
        candidate = trimmed[: last_obj + 1]
        # Remove trailing comma if present
        candidate = re.sub(r",\s*$", "", candidate)
        # Ensure intents array closure
        if '"intents"' in candidate and not candidate.rstrip().endswith("}]}"):
            # If array not closed, append ]}
            # Detect if we are inside array (presence of '[{' without closing ])
            if "[{" in candidate and not re.search(r"\}\s*\]", candidate):
                candidate = candidate + "]}"
            elif candidate.count("{") > candidate.count("}"):
                # Balance braces crudely
                missing = candidate.count("{") - candidate.count("}")
                candidate = candidate + ("}" * missing)
        return candidate

    async def generate_title(
        self,
        messages: List[Message],
        circuit_breaker: Optional[CircuitBreakerConfig] = None,
    ) -> str:
        """
        Generate a concise, descriptive title for a conversation based on its messages.

        Args:
            messages: List of conversation messages to analyze
            circuit_breaker: Optional circuit breaker configuration

        Returns:
            str: Generated conversation title (2-6 words)

        Raises:
            IntentAnalysisError: When title generation fails
        """
        try:
            # Extract text from all messages for context
            conversation_text = ""
            for message in messages[-5:]:  # Use last 5 messages for context
                text = extract_message_text(message)
                if text.strip():
                    role = "User" if message.role.value == "user" else "Assistant"
                    conversation_text += f"{role}: {text}\n"

            if not conversation_text.strip():
                return "New Conversation"

            title_prompt = f"""
Generate a concise, descriptive title for this conversation. The title should:
- Be 2-6 words maximum
- Capture the main topic or purpose
- Be clear and professional
- Not include quotes or special characters
- Be suitable as a conversation label

Conversation:
{conversation_text}

Title:"""

            # Use pipeline with title generation
            with self.pipeline_factory.pipeline(
                self.profile, str, PipelinePriority.MEDIUM, circuit_breaker
            ) as pipeline:
                from runner import (  # pylint: disable=import-outside-toplevel
                    run_pipeline,
                )

                result = await run_pipeline(
                    messages=title_prompt,
                    pipeline=pipeline,
                    tools=None,
                    grammar=None,
                )

                if not result or not result.message:
                    return "Untitled Conversation"

                title = extract_message_text(result.message).strip()

                # Clean up the title
                title = title.replace('"', "").replace("'", "").strip()

                # Ensure it's not too long
                words = title.split()
                if len(words) > 6:
                    title = " ".join(words[:6])

                # Fallback if empty
                if not title:
                    title = "New Conversation"

                composer_logger.logger.info(
                    "Title generated successfully",
                    extra={
                        "title": title,
                        "word_count": len(title.split()),
                        "message_count": len(messages),
                    },
                )

                return title

        except Exception as e:
            composer_logger.log_error(e, {"context": "title_generation"})
            # Provide fallback title instead of raising error
            return "Conversation"

    # All classification methods are now handled by LLM analysis above
