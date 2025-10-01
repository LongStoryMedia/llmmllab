"""
Intent classification node for workflow routing.
Analyzes user intent to determine workflow routing, RAG depth, and tool requirements.
"""

from typing import List

from models.lang_chain_message import LangChainMessage
from models.intent_analysis import IntentAnalysis
from models.complexity_level import ComplexityLevel
from models.required_capability import RequiredCapability
from models.computational_requirement import ComputationalRequirement
from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError

# Lazy imports to avoid circular dependencies
from db import storage  # pylint: disable=import-outside-toplevel
from utils.grammar_generator import get_grammar_for_model  # pylint: disable=import-outside-toplevel


class IntentClassifierNode:
    """
    Intent classification agent with grammar-constrained structured output.
    
    Analyzes user intent to determine workflow routing, RAG depth, and tool requirements.
    Implements structured output validation using IntentAnalysis schema.
    """

    def __init__(self, pipeline_factory):
        """
        Initialize intent classifier node.
        
        Args:
            pipeline_factory: Factory for creating structured pipelines
        """
        self.pipeline_factory = pipeline_factory
        self.logger = composer_logger.logger

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Analyze user intent with guaranteed structured output.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with intent classification
        """
        try:
            if not state.messages:
                return state

            user_id = getattr(state, 'user_id', None)
            if not user_id:
                raise NodeExecutionError("intent_classifier", Exception("User ID required for intent classification"))

            # Get latest user message
            user_messages = [msg for msg in state.messages if getattr(msg, 'role', 'user') == "user"]
            if not user_messages:
                return state

            latest_query = getattr(user_messages[-1], 'content', '')

            # User configuration and model profile will be accessed by pipeline factory internally

            self.logger.info(
                "Analyzing user intent",
                extra={
                    "user_id": user_id,
                    "query_length": len(latest_query)
                }
            )

            # Create grammar-constrained pipeline for intent analysis
            structured_pipeline = await self.pipeline_factory.create_structured_pipeline(
                prompt_template=self._get_intent_prompt(),
                output_schema=IntentAnalysis,
                grammar=get_grammar_for_model(IntentAnalysis),
                enable_fallback=True
            )

            # Execute intent analysis with structured output
            intent_analysis = await structured_pipeline.execute({
                "user_query": latest_query,
                "conversation_context": self._format_context(state.messages)
            })

            # Update state with validated structured output
            state.intent_classification = intent_analysis
            # Map complexity level to RAG depth for backward compatibility
            if hasattr(intent_analysis, 'complexity_level'):
                if intent_analysis.complexity_level in [ComplexityLevel.COMPLEX, ComplexityLevel.SPECIALIZED]:
                    state.rag_depth_config = "DEEP"
                else:
                    state.rag_depth_config = "SHALLOW"
            else:
                state.rag_depth_config = "SHALLOW"

            self.logger.info(
                "Intent classification completed",
                extra={
                    "user_id": user_id,
                    "primary_intent": intent_analysis.primary_intent,
                    "rag_depth": state.rag_depth_config,
                    "complexity_level": intent_analysis.complexity_level.value
                }
            )

            return state

        except Exception as e:
            self.logger.error(
                "Intent classification failed",
                extra={
                    "user_id": getattr(state, 'user_id', 'unknown'),
                    "error": str(e)
                }
            )
            
            # Create fallback intent analysis with correct fields
            fallback_intent = IntentAnalysis(
                primary_intent="general_chat",
                complexity_level=ComplexityLevel.SIMPLE,
                required_capabilities=[RequiredCapability.TEXT_PROCESSING],
                computational_requirements=[ComputationalRequirement.HIGH_MEMORY],
                domain_specificity=0.1,
                reusability_potential=0.8,
                confidence=0.1
            )
            
            state.intent_classification = fallback_intent
            state.rag_depth_config = "SHALLOW"
            
            return state

    def _get_intent_prompt(self) -> str:
        """Get the intent classification prompt template."""
        return """Analyze the user's request and classify their intent.

User Query: {user_query}
Conversation Context: {conversation_context}

Classify the intent including:
1. Main intent category (research, creative, technical_help, general_chat, etc.)
2. Required complexity level (LOW, MEDIUM, HIGH, VERY_HIGH)
3. Required capabilities needed to fulfill the request
4. Computational requirements (memory, processing power)
5. Domain specificity score (0.0 to 1.0)
6. Reusability potential (0.0 to 1.0)
7. Confidence in classification (0.0 to 1.0)

Consider:
- Simple questions = LOW complexity
- Research topics, analysis, current events = HIGH complexity
- Calculations, data processing = MEDIUM complexity
- Creative tasks may need specialized capabilities

Respond with valid JSON matching the IntentAnalysis schema.
All fields are required and must conform to the specified enums and constraints."""

    def _format_context(self, messages: List[LangChainMessage]) -> str:
        """Format conversation context for intent analysis."""
        if len(messages) <= 1:
            return "No prior context"
        
        context_lines = []
        for msg in messages[-5:]:  # Last 5 messages for context
            role = getattr(msg, 'role', 'user').title()
            content = getattr(msg, 'content', '')[:150]  # Truncate for brevity
            context_lines.append(f"{role}: {content}")
        
        return "\n".join(context_lines)