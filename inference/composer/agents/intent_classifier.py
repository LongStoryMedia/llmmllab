"""
Intent analysis and classification agent.
Performs comprehensive intent analysis following the capability-driven architecture.
Maps user requests to RequiredCapabilities and assesses computational complexity.
"""
import asyncio
import re
from typing import Dict, Any, Optional, List, Set
import sys
sys.path.append('/Users/lons7862/workspace/llmmllab/inference')

from models.conversation_ctx import ConversationCtx
from models.intent_analysis import IntentAnalysis
from models.complexity_level import ComplexityLevel
from models.required_capability import RequiredCapability
from models.computational_requirement import ComputationalRequirement
from composer.monitoring.logging import composer_logger
from composer.core.errors import IntentAnalysisError


class IntentClassifierAgent:
    """
    LLM-based intent analysis for workflow routing and tool selection.
    
    The IntentClassifierAgent is mandated to execute early in the graph flow.
    Its structured output guides RAG depth selection and tool orchestration.
    """
    
    def __init__(self):
        # Initialize LLM for intent classification
        # This would typically use a lightweight, fast model for quick classification
        self.classification_model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the intent classification model."""
        try:
            # Placeholder for model initialization
            # In actual implementation, this would initialize a lightweight LLM
            # or use the existing pipeline infrastructure
            composer_logger.logger.info("Intent classifier initialized")
        except Exception as e:
            composer_logger.log_error(e, {"context": "intent_classifier_init"})
    
    async def analyze(self, conversation_ctx: ConversationCtx) -> IntentAnalysis:
        """
        Analyze conversation context to determine intent, complexity, and requirements.
        
        Follows the capability-driven architecture:
        User Request → IntentAnalysis → RequiredCapabilities → ModelProfileType → ModelTask
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Extract user query from conversation context
            user_query = self._extract_user_query(conversation_ctx)
            
            # Analyze primary intent
            primary_intent = self._classify_primary_intent(user_query)
            
            # Assess complexity level
            complexity_level = self._assess_complexity(user_query)
            
            # Identify required capabilities
            required_capabilities = self._identify_required_capabilities(user_query, primary_intent)
            
            # Extract computational requirements
            computational_requirements = self._extract_computational_requirements(
                user_query, complexity_level, required_capabilities
            )
            
            # Calculate domain specificity score
            domain_specificity = self._calculate_domain_specificity(user_query, primary_intent)
            
            # Calculate reusability potential
            reusability_potential = self._calculate_reusability_potential(
                user_query, complexity_level, required_capabilities
            )
            
            # Calculate confidence in analysis
            confidence = self._calculate_confidence(
                user_query, complexity_level, required_capabilities, computational_requirements
            )
            
            # Create structured intent analysis
            intent_analysis = IntentAnalysis(
                primary_intent=primary_intent,
                complexity_level=complexity_level,
                required_capabilities=list(required_capabilities),
                computational_requirements=list(computational_requirements),
                domain_specificity=domain_specificity,
                reusability_potential=reusability_potential,
                confidence=confidence
            )
            
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            composer_logger.log_intent_analysis(
                intent_result={
                    'primary_intent': primary_intent,
                    'complexity': complexity_level.value,
                    'capabilities_count': len(required_capabilities)
                },
                confidence=confidence,
                processing_time_ms=processing_time
            )
            
            return intent_analysis
            
        except Exception as e:
            composer_logger.log_error(e, {"context": "intent_analysis"})
            raise IntentAnalysisError(f"Intent analysis failed: {e}")
    
    def _extract_user_query(self, conversation_ctx: ConversationCtx) -> str:
        """Extract the user query from conversation context."""
        if not conversation_ctx.messages:
            return ""
        
        # Find the last user message
        for message in reversed(conversation_ctx.messages):
            if message.role.value == 'user':
                # Handle both string content and MessageContent list
                if isinstance(message.content, str):
                    return message.content
                elif isinstance(message.content, list) and len(message.content) > 0:
                    # Extract text from first content item
                    first_content = message.content[0]
                    if first_content.text:
                        return first_content.text
                    else:
                        return str(first_content.type.value) if first_content.type else ""
                return str(message.content)
        
        return ""
    
    def _classify_primary_intent(self, user_query: str) -> str:
        """Classify the primary intent from available enum values."""
        query_lower = user_query.lower()
        
        # Intent classification based on keywords
        if any(keyword in query_lower for keyword in ['research', 'investigate', 'study', 'analyze', 'examine']):
            return 'research'
        elif any(keyword in query_lower for keyword in ['create', 'generate', 'write', 'compose', 'design', 'make']):
            return 'creative'
        elif any(keyword in query_lower for keyword in ['code', 'program', 'debug', 'implement', 'develop', 'algorithm']):
            return 'technical'
        elif any(keyword in query_lower for keyword in ['analyze', 'evaluation', 'assessment', 'breakdown']):
            return 'analysis'
        elif any(keyword in query_lower for keyword in ['summarize', 'summary', 'condense', 'brief']):
            return 'summarization'
        elif any(keyword in query_lower for keyword in ['remember', 'recall', 'previous', 'earlier', 'before']):
            return 'memory_retrieval'
        elif any(keyword in query_lower for keyword in ['search', 'find', 'look up', 'web']):
            return 'web_search'
        elif any(keyword in query_lower for keyword in ['image', 'picture', 'photo', 'visual', 'draw']):
            return 'image_generation'
        elif any(keyword in query_lower for keyword in ['data', 'dataset', 'process', 'transform', 'csv', 'json']):
            return 'data_processing'
        else:
            return 'chat'
    
    def _assess_complexity(self, user_query: str) -> ComplexityLevel:
        """Assess the complexity level of the user request."""
        query_lower = user_query.lower()
        query_length = len(user_query)
        
        # Specialized complexity indicators
        specialized_keywords = ['algorithm', 'optimization', 'machine learning', 'neural network', 'quantum', 'cryptography']
        if any(keyword in query_lower for keyword in specialized_keywords):
            return ComplexityLevel.SPECIALIZED
        
        # Complex indicators
        complex_keywords = ['comprehensive', 'detailed analysis', 'in-depth', 'thorough', 'complete']
        if any(keyword in query_lower for keyword in complex_keywords) or query_length > 200:
            return ComplexityLevel.COMPLEX
        
        # Moderate indicators
        moderate_keywords = ['analyze', 'compare', 'research', 'investigate']
        if any(keyword in query_lower for keyword in moderate_keywords) or query_length > 100:
            return ComplexityLevel.MODERATE
        
        # Simple indicators
        if query_length > 20:
            return ComplexityLevel.SIMPLE
        
        # Trivial for very short queries
        return ComplexityLevel.TRIVIAL
    
    def _identify_required_capabilities(self, user_query: str, primary_intent: str) -> Set[RequiredCapability]:
        """Identify what capabilities are required for the request."""
        capabilities = set()
        query_lower = user_query.lower()
        
        # Intent-based capability mapping
        intent_capability_map = {
            'research': {RequiredCapability.WEB_SEARCH, RequiredCapability.INFORMATION_RETRIEVAL, RequiredCapability.REASONING},
            'creative': {RequiredCapability.TEXT_PROCESSING, RequiredCapability.REASONING, RequiredCapability.GENERAL_KNOWLEDGE},
            'technical': {RequiredCapability.REASONING, RequiredCapability.TEXT_PROCESSING, RequiredCapability.GENERAL_KNOWLEDGE},
            'analysis': {RequiredCapability.REASONING, RequiredCapability.TEXT_PROCESSING, RequiredCapability.GENERAL_KNOWLEDGE},
            'summarization': {RequiredCapability.SUMMARIZATION, RequiredCapability.TEXT_PROCESSING},
            'memory_retrieval': {RequiredCapability.CONVERSATION_MEMORY, RequiredCapability.INFORMATION_RETRIEVAL},
            'web_search': {RequiredCapability.WEB_SEARCH, RequiredCapability.INFORMATION_RETRIEVAL},
            'image_generation': {RequiredCapability.TEXT_PROCESSING, RequiredCapability.GENERAL_KNOWLEDGE},
            'data_processing': {RequiredCapability.TEXT_PROCESSING, RequiredCapability.REASONING},
            'chat': {RequiredCapability.GENERAL_KNOWLEDGE, RequiredCapability.TEXT_PROCESSING}
        }
        
        # Add capabilities based on primary intent
        capabilities.update(intent_capability_map.get(primary_intent, set()))
        
        # Keyword-based capability detection
        if any(keyword in query_lower for keyword in ['calculate', 'math', 'compute', 'equation']):
            capabilities.add(RequiredCapability.BASIC_MATH)
        
        if any(keyword in query_lower for keyword in ['search', 'find', 'lookup', 'web']):
            capabilities.add(RequiredCapability.WEB_SEARCH)
        
        if any(keyword in query_lower for keyword in ['remember', 'previous', 'earlier', 'before']):
            capabilities.add(RequiredCapability.CONVERSATION_MEMORY)
        
        # Ensure at least one capability
        if not capabilities:
            capabilities.add(RequiredCapability.GENERAL_KNOWLEDGE)
        
        return capabilities
    
    def _extract_computational_requirements(
        self, 
        user_query: str, 
        complexity_level: ComplexityLevel, 
        required_capabilities: Set[RequiredCapability]
    ) -> Set[ComputationalRequirement]:
        """Extract computational requirements based on query analysis."""
        requirements = set()
        query_lower = user_query.lower()
        
        # Complexity-based requirements
        if complexity_level in [ComplexityLevel.COMPLEX, ComplexityLevel.SPECIALIZED]:
            requirements.add(ComputationalRequirement.COMPLEX_REASONING)
        
        # Capability-based requirements
        if RequiredCapability.WEB_SEARCH in required_capabilities:
            requirements.add(ComputationalRequirement.EXTERNAL_API_CALLS)
        
        if len(required_capabilities) > 3:
            requirements.add(ComputationalRequirement.PARALLEL_PROCESSING)
        
        # Content-based detection
        if any(keyword in query_lower for keyword in ['large', 'big data', 'massive', 'huge']):
            requirements.add(ComputationalRequirement.LARGE_DATA_HANDLING)
            requirements.add(ComputationalRequirement.HIGH_MEMORY)
        
        if any(keyword in query_lower for keyword in ['image', 'video', 'audio', 'multimodal']):
            requirements.add(ComputationalRequirement.MULTI_MODAL_PROCESSING)
            requirements.add(ComputationalRequirement.GPU_ACCELERATION)
        
        if any(keyword in query_lower for keyword in ['real-time', 'live', 'instant', 'immediate']):
            requirements.add(ComputationalRequirement.REAL_TIME_PROCESSING)
        
        if any(keyword in query_lower for keyword in ['file', 'document', 'save', 'export']):
            requirements.add(ComputationalRequirement.FILE_OPERATIONS)
        
        if any(keyword in query_lower for keyword in ['database', 'sql', 'query', 'table']):
            requirements.add(ComputationalRequirement.DATABASE_OPERATIONS)
        
        return requirements
    
    def _calculate_domain_specificity(self, user_query: str, primary_intent: str) -> float:
        """Calculate domain specificity score (0-1)."""
        query_lower = user_query.lower()
        
        # Domain-specific keywords
        domain_keywords = [
            'medical', 'legal', 'financial', 'scientific', 'academic', 'technical',
            'engineering', 'research', 'clinical', 'pharmaceutical', 'biotechnology',
            'quantum', 'neural', 'machine learning', 'ai', 'cryptocurrency', 'blockchain'
        ]
        
        # Count domain-specific terms
        domain_matches = sum(1 for keyword in domain_keywords if keyword in query_lower)
        
        # Base score from intent
        intent_scores = {
            'technical': 0.7,
            'research': 0.6,
            'analysis': 0.5,
            'data_processing': 0.6,
            'creative': 0.3,
            'chat': 0.1
        }
        
        base_score = intent_scores.get(primary_intent, 0.2)
        
        # Adjust for domain matches
        domain_boost = min(domain_matches * 0.2, 0.6)
        
        return min(base_score + domain_boost, 1.0)
    
    def _calculate_reusability_potential(
        self, 
        user_query: str, 
        complexity_level: ComplexityLevel, 
        required_capabilities: Set[RequiredCapability]
    ) -> float:
        """Calculate reusability potential score (0-1)."""
        query_lower = user_query.lower()
        
        # Base score from complexity
        complexity_scores = {
            ComplexityLevel.TRIVIAL: 0.2,
            ComplexityLevel.SIMPLE: 0.4,
            ComplexityLevel.MODERATE: 0.6,
            ComplexityLevel.COMPLEX: 0.8,
            ComplexityLevel.SPECIALIZED: 0.9
        }
        
        base_score = complexity_scores.get(complexity_level, 0.5)
        
        # Adjust for capability diversity
        capability_boost = min(len(required_capabilities) * 0.1, 0.3)
        
        # Reduce for highly personal/specific queries
        personal_keywords = ['my', 'me', 'i', 'personal', 'private', 'specific to me']
        personal_penalty = sum(0.1 for keyword in personal_keywords if keyword in query_lower)
        
        score = base_score + capability_boost - personal_penalty
        return max(0.1, min(score, 1.0))
    
    def _calculate_confidence(
        self, 
        user_query: str, 
        complexity_level: ComplexityLevel, 
        required_capabilities: Set[RequiredCapability], 
        computational_requirements: Set[ComputationalRequirement]
    ) -> float:
        """Calculate confidence in the analysis (0-1)."""
        base_confidence = 0.7
        
        # Adjust based on query length and clarity
        query_length = len(user_query)
        if query_length < 10:
            base_confidence -= 0.2  # Very short queries are ambiguous
        elif query_length > 200:
            base_confidence -= 0.1  # Very long queries may be unclear
        else:
            base_confidence += 0.1  # Good length for analysis
        
        # Adjust based on complexity clarity
        if complexity_level in [ComplexityLevel.TRIVIAL, ComplexityLevel.SPECIALIZED]:
            base_confidence += 0.1  # Clear extremes
        
        # Adjust based on capability identification
        if len(required_capabilities) == 1:
            base_confidence += 0.1  # Clear single capability
        elif len(required_capabilities) > 5:
            base_confidence -= 0.1  # Too many may indicate confusion
        
        # Adjust based on computational requirements specificity
        if len(computational_requirements) > 0:
            base_confidence += 0.05
        
        return max(0.1, min(base_confidence, 1.0))
    
    def determine_rag_depth(self, intent_analysis: IntentAnalysis) -> str:
        """Determine RAG depth based on intent analysis for backward compatibility."""
        # Map new architecture to legacy RAG depth decisions
        if (intent_analysis.complexity_level in [ComplexityLevel.COMPLEX, ComplexityLevel.SPECIALIZED] or
            intent_analysis.primary_intent in ['research', 'technical', 'analysis'] or
            RequiredCapability.WEB_SEARCH in intent_analysis.required_capabilities):
            return 'DEEP'
        else:
            return 'SHALLOW'
    
    async def decide_search_depth(self, conversation_ctx: ConversationCtx) -> str:
        """Legacy method for RAG depth decision - delegates to new architecture."""
        intent_analysis = await self.analyze(conversation_ctx)
        return self.determine_rag_depth(intent_analysis)