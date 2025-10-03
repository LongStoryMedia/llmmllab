"""
Intent classification node for workflow routing.
Wraps the IntentClassifierAgent to provide LangGraph workflow integration.
"""

from models.intent_analysis import IntentAnalysis
from models.complexity_level import ComplexityLevel
from models.required_capability import RequiredCapability
from models.computational_requirement import ComputationalRequirement
from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError


class IntentClassifierNode:
    """
    LangGraph node wrapper for intent classification.
    
    Wraps the IntentClassifierAgent to provide workflow state integration and 
    proper LangGraph node interface. Handles state updates and RAG routing configuration.
    """

    def __init__(self):
        """
        Initialize intent classifier node with agent delegation.
        """
        # Lazy import to avoid circular dependencies
        from composer.agents.intent_classifier import IntentClassifierAgent  # pylint: disable=import-outside-toplevel
        
        self.agent = IntentClassifierAgent()
        self.logger = composer_logger.logger

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Execute intent classification using the wrapped agent.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with intent classification and RAG config
        """
        try:
            user_id = getattr(state, 'user_id', None)
            if not user_id:
                raise NodeExecutionError("intent_classifier", Exception("User ID required for intent classification"))

            if not state.messages:
                return state

            self.logger.info(
                "Intent classifier node executing",
                extra={"user_id": user_id, "message_count": len(state.messages)}
            )

            # Convert WorkflowState messages to Message format expected by agent
            messages = []
            for msg in state.messages:
                # Convert LangChainMessage to Message format if needed
                messages.append(msg)

            # Delegate to the specialized intent classifier agent
            intent_analysis = await self.agent.analyze(user_id, messages)

            # Update workflow state with analysis results
            state.intent_classification = intent_analysis
            
            # Configure RAG depth based on intent analysis
            state.search_depth_config = self.agent.determine_search_depth(intent_analysis).upper()

            self.logger.info(
                "Intent classification completed",
                extra={
                    "user_id": user_id,
                    "primary_intent": intent_analysis.primary_intent,
                    "search_depth": state.search_depth_config,
                    "complexity_level": intent_analysis.complexity_level.value
                }
            )

            return state

        except Exception as e:
            self.logger.error(
                "Intent classifier node failed",
                extra={
                    "user_id": getattr(state, 'user_id', 'unknown'),
                    "error": str(e)
                }
            )
            
            # Use agent's fallback mechanism
            try:
                fallback_intent = self.agent._fallback_heuristic_analysis(
                    getattr(state.messages[-1], 'content', '') if state.messages else "unknown query"
                )
                state.intent_classification = fallback_intent
                state.search_depth_config = "SHALLOW"
            except Exception:
                # Final fallback if even the heuristic fails
                minimal_intent = IntentAnalysis(
                    primary_intent="chat",
                    complexity_level=ComplexityLevel.SIMPLE,
                    required_capabilities=[RequiredCapability.TEXT_PROCESSING],
                    computational_requirements=[ComputationalRequirement.COMPLEX_REASONING],
                    domain_specificity=0.1,
                    reusability_potential=0.8,
                    confidence=0.1
                )
                state.intent_classification = minimal_intent
                state.search_depth_config = "SHALLOW"
            
            return state