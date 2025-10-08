"""
Response Format Analysis Node for LangGraph workflow integration.
Determines appropriate response format and technical domain based on sophisticated analysis.
"""

import json
from typing import Optional

from pydantic import BaseModel

from composer.graph.state import WorkflowState
from composer.nodes.base_node import BaseNode
from composer.agents.engineering_agent import TechnicalDomain, ResponseFormat
from composer.utils.conversion import langchain_message_to_message
from utils.model_profile import get_model_profile_for_task
from utils.message import extract_message_text
from models import IntentAnalysis, UserConfig, ModelProfileType, PipelinePriority
from models.default_configs import DEFAULT_CIRCUIT_BREAKER_CONFIG
from runner import PipelineFactory


class ResponseFormatAnalysisNode(BaseNode):
    """
    LangGraph node for determining response format and technical domain.

    Uses sophisticated analysis rather than simple keyword matching to determine
    the most appropriate response format and technical domain for engineering responses.
    This node should be placed after intent analysis and before engineering response generation.
    """

    def __init__(self, pipeline_factory: PipelineFactory):
        """
        Initialize response format analysis node.

        Args:
            pipeline_factory: Factory for creating structured pipelines
        """
        super().__init__(
            "ResponseFormatAnalysisNode", pipeline_factory=pipeline_factory
        )

    def _initialize_node(self, pipeline_factory: PipelineFactory, **kwargs):
        """Initialize node-specific components."""
        self.pipeline_factory = pipeline_factory

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Analyze user request to determine response format and technical domain.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with response_format and technical_domain set
        """
        try:
            assert state.user_id
            assert state.current_user_message
            # Validate state requirements and user ID
            self._validate_state_requirements(
                state, require_messages=True, require_intent_classification=True
            )
            user_id = state.user_id

            # Extract user query from most recent user message using langgraph utility
            user_query = extract_message_text(
                langchain_message_to_message(state.current_user_message)
            )

            if not user_query.strip():
                return state

            self.logger.info(
                "Analyzing response format and technical domain",
                extra={
                    "user_id": user_id,
                    "query_length": len(user_query),
                },
            )

            assert (
                state.user_config is not None
            ), "UserConfig must be available in state"
            assert (
                state.intent_classification is not None
            ), "IntentAnalysis must be available in state"

            # Use LLM-based analysis to determine response format
            for intent in state.intent_classification:
                await self._analyze_response_format(
                    user_query, state.user_config, intent
                )
                await self._analyze_technical_domain(
                    user_query, state.user_config, intent
                )

                self.logger.info(
                    "Response format analysis completed",
                    extra={
                        "user_id": user_id,
                        "intent": intent.primary_intent,
                        "response_format": intent.response_format,
                        "technical_domain": intent.technical_domain,
                    },
                )

            return state

        except Exception as e:
            self.logger.error(
                "Response format analysis failed",
                extra={
                    "user_id": getattr(state, "user_id", "unknown"),
                    "error": str(e),
                },
            )
            return state

    async def _analyze_response_format(
        self,
        user_query: str,
        user_config: UserConfig,
        intent_analysis: Optional[IntentAnalysis],
    ) -> ResponseFormat:
        """
        Use LLM-based analysis to determine the most appropriate response format.

        Args:
            user_query: The user's query text
            intent_analysis: Intent classification results

        Returns:
            Appropriate response format based on sophisticated analysis
        """
        # Create analysis prompt for response format
        analysis_prompt = f"""Analyze this user request and determine the most appropriate response format.

User Query: {user_query}

Primary Intent: {intent_analysis.primary_intent if intent_analysis else 'unknown'}
Complexity Level: {intent_analysis.complexity_level if intent_analysis else 'unknown'}

Available Response Formats:
1. CODE_SOLUTION - For requests asking for specific code implementations
2. STEP_BY_STEP_GUIDE - For requests asking how to do something or tutorials
3. BEST_PRACTICES - For requests about recommendations, approaches, or best practices
4. TROUBLESHOOTING - For debugging, fixing errors, or solving problems
5. DETAILED_ANALYSIS - For analytical questions or comprehensive explanations

Consider:
- What type of output would be most helpful to the user?
- Are they asking for implementation details, guidance, recommendations, or analysis?
- What format would best serve their apparent goal?

Respond with only the format name (e.g., CODE_SOLUTION)."""

        try:
            model_profile = await get_model_profile_for_task(
                user_config.model_profiles,
                ModelProfileType.Analysis,
                user_config.user_id,
            )
            cb = (
                user_config.circuit_breaker
                if user_config.circuit_breaker
                else DEFAULT_CIRCUIT_BREAKER_CONFIG
            )

            class _ResFmt(BaseModel):
                format: ResponseFormat

            from runner import run_pipeline  # pylint: disable=import-outside-toplevel

            with self.pipeline_factory.pipeline(
                model_profile, str, PipelinePriority.HIGH, cb
            ) as pipe:
                result = await run_pipeline(
                    analysis_prompt, pipeline=pipe, grammar=_ResFmt
                )
                return (
                    _ResFmt(**(json.loads(extract_message_text(result.message)))).format
                    if result and result.message
                    else ResponseFormat.DETAILED_ANALYSIS
                )

        except Exception as e:
            self.logger.warning(
                f"LLM-based format analysis failed, using fallback: {e}"
            )
            # Fallback to intent-based analysis
            return self._determine_response_format_from_intent(intent_analysis)

    async def _analyze_technical_domain(
        self,
        user_query: str,
        user_config: UserConfig,
        intent_analysis: Optional[IntentAnalysis],
    ) -> TechnicalDomain:
        """
        Use LLM-based analysis to determine the most appropriate technical domain.

        Args:
            user_query: The user's query text
            intent_analysis: Intent classification results

        Returns:
            Appropriate technical domain based on sophisticated analysis
        """
        # Create analysis prompt for technical domain
        analysis_prompt = f"""Analyze this user request and determine the most appropriate technical domain.

User Query: {user_query}

Primary Intent: {intent_analysis.primary_intent if intent_analysis else 'unknown'}

Available Technical Domains:
1. SOFTWARE_DEVELOPMENT - Programming, coding, software engineering
2. SYSTEM_ARCHITECTURE - System design, architecture, infrastructure planning
3. DATA_ENGINEERING - Data pipelines, databases, data processing
4. DEVOPS_INFRASTRUCTURE - Deployment, CI/CD, infrastructure management
5. SECURITY_ENGINEERING - Security, authentication, encryption
6. MACHINE_LEARNING - AI, ML models, data science
7. GENERAL_ENGINEERING - Other engineering disciplines or general engineering questions

Consider:
- What technical field does this question primarily belong to?
- What expertise would be most relevant for answering this question?
- Which domain would provide the most specialized and helpful response?

Respond with only the domain name (e.g., SOFTWARE_DEVELOPMENT)."""

        try:
            model_profile = await get_model_profile_for_task(
                user_config.model_profiles,
                ModelProfileType.Analysis,
                user_config.user_id,
            )
            cb = (
                user_config.circuit_breaker
                if user_config.circuit_breaker
                else DEFAULT_CIRCUIT_BREAKER_CONFIG
            )

            class _ResFmt(BaseModel):
                domain: TechnicalDomain

            from runner import run_pipeline  # pylint: disable=import-outside-toplevel

            with self.pipeline_factory.pipeline(
                model_profile, str, PipelinePriority.HIGH, cb
            ) as pipe:
                result = await run_pipeline(
                    analysis_prompt, pipeline=pipe, grammar=_ResFmt
                )
                return (
                    _ResFmt(**(json.loads(extract_message_text(result.message)))).domain
                    if result and result.message
                    else TechnicalDomain.GENERAL_ENGINEERING
                )

        except Exception as e:
            self.logger.warning(
                f"LLM-based domain analysis failed, using fallback: {e}"
            )
            # Fallback to intent-based analysis
            return self._determine_technical_domain_from_intent(intent_analysis)

    def _determine_response_format_from_intent(self, intent_analysis) -> ResponseFormat:
        """
        Fallback method to determine response format from intent analysis.

        Args:
            intent_analysis: Intent classification results

        Returns:
            Appropriate response format
        """
        if not intent_analysis:
            return ResponseFormat.DETAILED_ANALYSIS

        primary_intent = str(intent_analysis.primary_intent).lower()

        # Map intents to response formats
        if any(
            keyword in primary_intent
            for keyword in ["code", "implement", "build", "create"]
        ):
            return ResponseFormat.CODE_SOLUTION
        elif any(
            keyword in primary_intent
            for keyword in ["how", "steps", "guide", "tutorial"]
        ):
            return ResponseFormat.STEP_BY_STEP_GUIDE
        elif any(
            keyword in primary_intent
            for keyword in ["best", "practice", "recommend", "approach"]
        ):
            return ResponseFormat.BEST_PRACTICES
        elif any(
            keyword in primary_intent
            for keyword in ["debug", "fix", "error", "issue", "problem"]
        ):
            return ResponseFormat.TROUBLESHOOTING
        else:
            return ResponseFormat.DETAILED_ANALYSIS

    def _determine_technical_domain_from_intent(
        self, intent_analysis
    ) -> TechnicalDomain:
        """
        Fallback method to determine technical domain from intent analysis.

        Args:
            intent_analysis: Intent classification results

        Returns:
            Appropriate technical domain
        """
        if not intent_analysis:
            return TechnicalDomain.GENERAL_ENGINEERING

        primary_intent = str(intent_analysis.primary_intent).lower()

        # Map common intents to technical domains
        domain_mapping = {
            "code": TechnicalDomain.SOFTWARE_DEVELOPMENT,
            "software": TechnicalDomain.SOFTWARE_DEVELOPMENT,
            "programming": TechnicalDomain.SOFTWARE_DEVELOPMENT,
            "architecture": TechnicalDomain.SYSTEM_ARCHITECTURE,
            "system": TechnicalDomain.SYSTEM_ARCHITECTURE,
            "design": TechnicalDomain.SYSTEM_ARCHITECTURE,
            "data": TechnicalDomain.DATA_ENGINEERING,
            "database": TechnicalDomain.DATA_ENGINEERING,
            "pipeline": TechnicalDomain.DATA_ENGINEERING,
            "deploy": TechnicalDomain.DEVOPS_INFRASTRUCTURE,
            "infrastructure": TechnicalDomain.DEVOPS_INFRASTRUCTURE,
            "devops": TechnicalDomain.DEVOPS_INFRASTRUCTURE,
            "security": TechnicalDomain.SECURITY_ENGINEERING,
            "ml": TechnicalDomain.MACHINE_LEARNING,
            "ai": TechnicalDomain.MACHINE_LEARNING,
            "model": TechnicalDomain.MACHINE_LEARNING,
        }

        for keyword, domain in domain_mapping.items():
            if keyword in primary_intent:
                return domain

        return TechnicalDomain.GENERAL_ENGINEERING
