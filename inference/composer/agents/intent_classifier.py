"""
Intent analysis and classification agent.
Performs comprehensive intent analysis following the capability-driven architecture.
Maps user requests to RequiredCapabilities and assesses computational complexity.
"""


from typing import List

from pydantic import BaseModel

from models import (
    IntentAnalysis,
    ComplexityLevel,
    RequiredCapability,
    ComputationalRequirement,
    Message,
)
from models.model_profile_type import ModelProfileType
from composer.monitoring.logging import composer_logger
from composer.core.errors import IntentAnalysisError
from runner.pipelines.run import run_pipeline
from utils.message import extract_message_text
from utils.grammar_generator import parse_structured_output
from utils.model_profile import get_model_profile


class IntentClassifierAgent:
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

    def __init__(self):
        """Initialize the intent classification agent."""
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
        self, user_id: str, current_user_message: Message
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

            # Get analysis model profile using shared utility (handles caching and user config access)
            try:
                mp = await get_model_profile(user_id, ModelProfileType.Analysis)
            except (ValueError, AssertionError) as e:
                composer_logger.logger.warning(
                    "Failed to get analysis model profile, using fallback",
                    extra={
                        "user_id": user_id,
                        "error": str(e),
                        "component": "intent_classifier",
                    },
                )
                # Return fallback intent analysis when model profile unavailable
                return [
                    IntentAnalysis(
                        primary_intent="chat",
                        complexity_level=ComplexityLevel.SIMPLE,
                        required_capabilities=[RequiredCapability.TEXT_PROCESSING],
                        computational_requirements=[
                            ComputationalRequirement.COMPLEX_REASONING
                        ],
                        domain_specificity=0.3,
                        reusability_potential=0.7,
                        confidence=0.6,
                    )
                ]

            # Use LLM for comprehensive intent analysis
            from runner import pipeline_factory

            # Use pipeline with default priority for intent analysis
            with pipeline_factory.pipeline(mp, str) as pipeline:
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
        # NOTE: Do NOT embed the raw JSON *schema* in the prompt (the model then echoes the schema
        # which breaks validation). Provide a clear natural language spec + minimal exemplar instead.
        analysis_prompt = f"""
You are an expert intent classification system. Analyze the user request and output ONLY JSON.

Enumerations (must use exactly these values where applicable):
    primary_intent: chat | research | creative | technical | summarization | analysis | tool_use | memory | embedding
    complexity_level: TRIVIAL | SIMPLE | MODERATE | COMPLEX | SPECIALIZED
    required_capabilities (array, choose relevant): basic_math, text_processing, information_retrieval, conversation_memory, web_search, summarization, reasoning, general_knowledge, api_integration, async_processing, file_manipulation, data_processing, image_processing, audio_processing, database_access, network_communication
    computational_requirements (array, choose relevant): high_memory, gpu_acceleration, parallel_processing, real_time_processing, large_data_handling, complex_reasoning, multi_modal_processing, external_api_calls, file_operations, database_operations

Instructions:
1. Decompose the request into logical sub-tasks ONLY if that materially helps execution; otherwise produce a single intent.
2. Produce one object per sub-task in the intents array.
3. domain_specificity, reusability_potential, confidence are floats 0.0–1.0.
4. Omit response_format / technical_domain unless clearly implied.
5. Output strictly valid JSON. No prose, no markdown.

User Request: {user_query}

Return JSON with this top-level structure ONLY:
{"intents": [
    {
        "primary_intent": "research",
        "complexity_level": "MODERATE",
        "required_capabilities": ["web_search", "reasoning"],
        "computational_requirements": ["complex_reasoning"],
        "domain_specificity": 0.4,
        "reusability_potential": 0.7,
        "confidence": 0.8
    }
]}

If multiple intents are needed, include additional objects in the intents array.
"""

        result = await run_pipeline(
            messages=analysis_prompt,
            pipeline=pipeline,
            tools=None,
            grammar=_Intnts,
        )

        txt = extract_message_text(result.message) if result and result.message else ""

        intents = parse_structured_output(txt, _Intnts)
        # Extract text from ChatResponse
        return intents.intents

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

    def _fallback_heuristic_analysis(self, user_query: str) -> IntentAnalysis:
        """
        Fallback heuristic analysis when grammar-constrained LLM fails.

        Provides basic intent classification using keyword matching when:
        - Grammar-constrained LLM analysis fails
        - Model profile unavailable
        - JSON parsing errors occur

        Args:
            user_query: User message text for heuristic analysis

        Returns:
            IntentAnalysis: Basic intent analysis with lower confidence scores
        """
        query_lower = user_query.lower()

        # Simple heuristic classification
        if any(kw in query_lower for kw in ["research", "investigate", "study"]):
            primary_intent = "research"
            capabilities = [RequiredCapability.WEB_SEARCH, RequiredCapability.REASONING]
        elif any(kw in query_lower for kw in ["create", "generate", "write"]):
            primary_intent = "creative"
            capabilities = [
                RequiredCapability.TEXT_PROCESSING,
                RequiredCapability.REASONING,
            ]
        elif any(kw in query_lower for kw in ["code", "program", "debug"]):
            primary_intent = "technical"
            capabilities = [
                RequiredCapability.REASONING,
                RequiredCapability.TEXT_PROCESSING,
            ]
        else:
            primary_intent = "chat"
            capabilities = [RequiredCapability.TEXT_PROCESSING]

        # Simple complexity assessment
        if len(user_query) > 200:
            complexity = ComplexityLevel.COMPLEX
        elif len(user_query) > 100:
            complexity = ComplexityLevel.MODERATE
        else:
            complexity = ComplexityLevel.SIMPLE

        return IntentAnalysis(
            primary_intent=primary_intent,
            complexity_level=complexity,
            required_capabilities=capabilities,
            computational_requirements=[],
            domain_specificity=0.3,
            reusability_potential=0.5,
            confidence=0.6,  # Lower confidence for heuristic fallback
        )

    # All classification methods are now handled by LLM analysis above
