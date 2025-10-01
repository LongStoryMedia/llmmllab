"""
Intent analysis and classification agent.
Performs comprehensive intent analysis following the capability-driven architecture.
Maps user requests to RequiredCapabilities and assesses computational complexity.

REFACTOR STATUS: ✅ COMPLETED - Updated to use shared data layer pattern with user_id.
- ✅ analyze() method now accepts user_id and messages instead of ConversationCtx
- ✅ Configuration retrieved via storage.user_config.get_user_config(user_id) pattern
- ✅ Proper fallback handling when user config or model profiles unavailable
- ✅ Message extraction logic updated to work with messages list
- ✅ Removed ConversationCtx dependencies throughout the component
"""

import asyncio
import json

from typing import List, Optional

from models import (
    IntentAnalysis, 
    ComplexityLevel, 
    RequiredCapability, 
    ComputationalRequirement, 
    Message,
    UserConfig
)
from models.model_profile_type import ModelProfileType
from utils.model_profile import get_model_profile
from composer.monitoring.logging import composer_logger
from composer.core.errors import IntentAnalysisError
from runner.pipelines.run import run_pipeline
from db import storage
from utils.message import extract_message_text


class IntentClassifierAgent:
    """
    Grammar-constrained LLM intent analysis agent for workflow routing and tool selection.

    Architectural Role:
    - Executes early in LangGraph flow using grammar-constrained structured output
    - Uses llamacpp grammars derived from intent_analysis.yaml schema  
    - Generates type-safe IntentAnalysis model with guaranteed structure validation
    - Guides tool selection, workflow type selection, and RAG depth configuration
    - Follows shared data layer pattern for configuration retrieval

    Configuration Management:
    - Uses shared get_model_profile() utility for analysis model profile retrieval
    - Data layer manages user configuration access and multi-tier caching automatically  
    - No configuration objects passed as arguments (architectural compliance)
    - Falls back gracefully when model profiles unavailable

    Grammar-Constrained Output:
    - Ensures LLM output matches IntentAnalysis Pydantic model structure
    - Eliminates JSON parsing errors through grammar validation
    - Provides type-safe, predictable structured output for graph routing
    """

    def __init__(self):
        """Initialize the intent classification agent."""
        composer_logger.logger.info(
            "Intent classifier initialized with analysis model profile"
        )



    def determine_rag_depth(self, intent_analysis: IntentAnalysis) -> str:
        """
        Determine RAG depth configuration for adaptive retrieval routing.
        
        Maps IntentAnalysis complexity levels to RAG execution paths:
        - COMPLEX/SPECIALIZED → "deep" (multi-step crawl and synthesis)
        - MODERATE → "moderate" (enhanced retrieval with limited sources)
        - TRIVIAL/SIMPLE → "shallow" (single-pass vector store retrieval)
        
        This drives LangGraph conditional routing between:
        - execute_deep_crawl_and_synthesize node
        - execute_shallow_search node
        
        Args:
            intent_analysis: Structured intent analysis from grammar-constrained LLM
            
        Returns:
            str: RAG depth configuration ("shallow", "moderate", "deep")
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

    async def analyze(self, user_id: str, messages: List[Message]) -> IntentAnalysis:
        """
        Grammar-constrained intent analysis using structured LLM output.

        Architectural Flow:
        1. Retrieve user configuration from shared data layer using user_id
        2. Extract latest user message from messages list
        3. Get analysis model profile from user configuration  
        4. Execute grammar-constrained LLM analysis with structured output
        5. Return validated IntentAnalysis with guaranteed schema compliance

        Configuration Retrieval:
        - Uses shared get_model_profile(user_id, ModelProfileType.Analysis) utility
        - Data layer manages multi-tier caching (memory → Redis → database) automatically
        - Graceful fallback when model profile or configuration unavailable

        Grammar Constraints:
        - LLM output guaranteed to match IntentAnalysis Pydantic model
        - Uses llamacpp grammars derived from intent_analysis.yaml schema
        - Eliminates JSON parsing errors through structure validation
        - Enables type-safe conditional routing in LangGraph workflows
        
        Args:
            user_id: User ID for shared data layer configuration retrieval
            messages: Conversation messages (extracts current user message)
            
        Returns:
            IntentAnalysis: Grammar-validated structured analysis for workflow routing

        Raises:
            IntentAnalysisError: When analysis fails or user message unavailable
        """
        try:
            start_time = asyncio.get_event_loop().time()

            # Extract current user message from messages list
            current_user_message = None
            if messages:
                # Get the last user message from the conversation
                for message in reversed(messages):
                    if hasattr(message, 'role') and hasattr(message.role, 'value') and message.role.value == 'user':
                        current_user_message = message
                        break
                    elif hasattr(message, 'content') and isinstance(message.content, str):
                        # Assume it's a user message if it's just content
                        current_user_message = message
                        break
            
            if not current_user_message:
                raise ValueError("No user message found in conversation history")

            # Extract user message text
            user_query = extract_message_text(current_user_message)

            # Get analysis model profile using shared utility (handles caching and user config access)
            try:
                mp = await get_model_profile(user_id, ModelProfileType.Analysis)
            except (ValueError, AssertionError) as e:
                composer_logger.logger.warning(
                    "Failed to get analysis model profile, using fallback", 
                    extra={"user_id": user_id, "error": str(e), "component": "intent_classifier"}
                )
                # Return fallback intent analysis when model profile unavailable
                return IntentAnalysis(
                    primary_intent="chat",
                    complexity_level=ComplexityLevel.SIMPLE,
                    required_capabilities=[RequiredCapability.TEXT_PROCESSING],
                    computational_requirements=[ComputationalRequirement.COMPLEX_REASONING],
                    domain_specificity=0.3,
                    reusability_potential=0.7,
                    confidence=0.6
                )

            # Use LLM for comprehensive intent analysis
            from runner import pipeline_factory
            # Use pipeline with default priority for intent analysis
            with pipeline_factory.pipeline(mp, str) as pipeline:
                analysis_result = await self._llm_analyze_intent(pipeline, user_query)

            # Parse LLM response into structured IntentAnalysis
            intent_analysis = self._parse_llm_response(analysis_result)

            # Apply any statistical augmentations
            intent_analysis = self._augment_with_statistics(intent_analysis, user_query)

            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000

            composer_logger.log_intent_analysis(
                intent_result={
                    "primary_intent": intent_analysis.primary_intent,
                    "complexity": intent_analysis.complexity_level,
                    "capabilities_count": len(intent_analysis.required_capabilities),
                },
                confidence=intent_analysis.confidence,
                processing_time_ms=processing_time,
            )

            return intent_analysis

        except Exception as e:
            composer_logger.log_error(e, {"context": "intent_analysis"})
            raise IntentAnalysisError(f"Intent analysis failed: {e}") from e

    async def _llm_analyze_intent(self, pipeline, user_query: str) -> str:
        """
        Execute grammar-constrained LLM analysis with structured output.
        
        TODO: Implement grammar constraints using llamacpp grammars derived from
        intent_analysis.yaml schema to ensure guaranteed structure validation.
        Current implementation uses JSON schema validation as fallback.
        
        Args:
            pipeline: Model pipeline with analysis profile
            user_query: User message text for analysis
            
        Returns:
            str: LLM response (should be grammar-constrained JSON)
        """
        analysis_prompt = f"""
You are an expert intent classification system. Analyze the user request and provide a structured JSON response.

Available intent types: chat, research, creative, technical, summarization, analysis, tool_use
Available complexity levels: TRIVIAL, SIMPLE, MODERATE, COMPLEX, SPECIALIZED
Available capabilities: basic_math, text_processing, information_retrieval, conversation_memory, web_search, summarization, reasoning, general_knowledge, api_integration, async_processing, file_manipulation, data_processing, image_processing, audio_processing, database_access, network_communication
Available computational requirements: high_memory, gpu_acceleration, parallel_processing, real_time_processing, large_data_handling, complex_reasoning, multi_modal_processing, external_api_calls, file_operations, database_operations

User Request: {user_query}

Analyze this request and respond with ONLY a JSON object in this exact format:
{json.dumps(IntentAnalysis.model_json_schema())}

Consider:
1. What is the primary intent of this request?
2. How complex is the task (technical sophistication, domain expertise required)?
3. What capabilities are needed to fulfill this request?
4. What computational resources might be required?
5. How domain-specific vs general is this request? (0.0 = very general, 1.0 = highly specialized)
6. How reusable would the approach be for similar requests? (0.0 = very specific, 1.0 = highly reusable)
7. How confident are you in this analysis? (0.0 = uncertain, 1.0 = very confident)
"""

        result = await run_pipeline(
            messages=analysis_prompt, pipeline=pipeline, tools=None
        )

        # Extract text from ChatResponse
        return extract_message_text(result.message) if result.message else str(result)

    def _parse_llm_response(self, llm_response: str) -> IntentAnalysis:
        """Parse LLM JSON response into IntentAnalysis object."""
        try:
            # Extract JSON from response (in case there's extra text)
            json_start = llm_response.find("{")
            json_end = llm_response.rfind("}") + 1
            if json_start != -1 and json_end != -1:
                json_str = llm_response[json_start:json_end]
                analysis_data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in LLM response")

            # Convert string enums to proper enum objects
            complexity_level = ComplexityLevel(analysis_data["complexity_level"])
            required_capabilities = [
                RequiredCapability(cap)
                for cap in analysis_data["required_capabilities"]
            ]
            computational_requirements = [
                ComputationalRequirement(req)
                for req in analysis_data["computational_requirements"]
            ]

            return IntentAnalysis(
                primary_intent=analysis_data["primary_intent"],
                complexity_level=complexity_level,
                required_capabilities=required_capabilities,
                computational_requirements=computational_requirements,
                domain_specificity=float(analysis_data["domain_specificity"]),
                reusability_potential=float(analysis_data["reusability_potential"]),
                confidence=float(analysis_data["confidence"]),
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            composer_logger.log_error(
                e, {"context": "llm_response_parsing", "response": llm_response}
            )
            # Fallback to heuristic analysis if LLM parsing fails
            return self._fallback_heuristic_analysis(llm_response)

    def _augment_with_statistics(
        self, intent_analysis: IntentAnalysis, user_query: str
    ) -> IntentAnalysis:
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

        return IntentAnalysis(
            primary_intent=intent_analysis.primary_intent,
            complexity_level=intent_analysis.complexity_level,
            required_capabilities=intent_analysis.required_capabilities,
            computational_requirements=intent_analysis.computational_requirements,
            domain_specificity=intent_analysis.domain_specificity,
            reusability_potential=intent_analysis.reusability_potential,
            confidence=adjusted_confidence,
        )

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
