"""
Intent analysis and classification agent.
Runs LLM-based intent classifiers early in workflows to set retrieval depth,
determine toolsets, and drive conditional routing.
"""
import asyncio
from typing import Dict, Any, Optional
import sys
sys.path.append('/Users/lons7862/workspace/llmmllab/inference')

from models.conversation_ctx import ConversationCtx
from models.intent_analysis import IntentAnalysis
from models.intent import Intent
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
        
        This is the primary method that outputs structured IntentAnalysis
        to guide subsequent workflow decisions.
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Extract relevant information from conversation context
            analysis_input = self._prepare_analysis_input(conversation_ctx)
            
            # Perform intent classification
            intent_result = await self._classify_intent(analysis_input)
            
            # Determine RAG depth based on complexity
            rag_depth = self._determine_rag_depth(intent_result)
            
            # Assess tool requirements
            tool_requirements = await self._assess_tool_requirements(intent_result, conversation_ctx)
            
            # Create structured intent analysis
            intent_analysis = IntentAnalysis(
                primary_intent=intent_result.get('primary_intent', 'chat'),
                secondary_intents=intent_result.get('secondary_intents', []),
                confidence=intent_result.get('confidence', 0.8),
                estimated_complexity=intent_result.get('complexity', 'medium'),
                requires_tools=tool_requirements['requires_tools'],
                requires_external_data=tool_requirements['requires_external_data'],
                rag_depth_recommendation=rag_depth,
                tool_specification=tool_requirements.get('tool_specification', '')
            )
            
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            composer_logger.log_intent_analysis(
                intent_result=intent_result,
                confidence=intent_analysis.confidence,
                processing_time_ms=processing_time
            )
            
            return intent_analysis
            
        except Exception as e:
            composer_logger.log_error(e, {"context": "intent_analysis"})
            raise IntentAnalysisError(f"Intent analysis failed: {e}")
    
    def _prepare_analysis_input(self, conversation_ctx: ConversationCtx) -> Dict[str, Any]:
        """Prepare input data for intent analysis."""
        # Extract the last few messages for context
        recent_messages = []
        if conversation_ctx.messages:
            recent_messages = conversation_ctx.messages[-5:]  # Last 5 messages
        
        # Extract user query (typically the last user message)
        user_query = ""
        for message in reversed(recent_messages):
            if message.role.value == 'user':
                user_query = message.content
                break
        
        return {
            "user_query": user_query,
            "message_count": len(conversation_ctx.messages) if conversation_ctx.messages else 0,
            "conversation_length": len(str(conversation_ctx.messages)) if conversation_ctx.messages else 0,
            "recent_messages": [{"role": msg.role.value, "content": msg.content[:200]} for msg in recent_messages],
            "user_config": conversation_ctx.user_config.dict() if conversation_ctx.user_config else {}
        }
    
    async def _classify_intent(self, analysis_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform the actual intent classification using LLM.
        
        In a full implementation, this would use a structured prompt to classify
        the user's intent into categories like: chat, research, creative, technical, etc.
        """
        user_query = analysis_input.get('user_query', '')
        
        # Simplified rule-based classification for now
        # In production, this would use an LLM with structured output
        intent_result = await self._rule_based_classification(user_query, analysis_input)
        
        return intent_result
    
    async def _rule_based_classification(self, user_query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simple rule-based classification as placeholder for LLM-based classification."""
        query_lower = user_query.lower()
        
        # Research indicators
        research_keywords = ['research', 'analyze', 'compare', 'investigate', 'study', 'examine', 'find out', 'search for']
        
        # Creative indicators  
        creative_keywords = ['create', 'generate', 'write', 'compose', 'design', 'make', 'build']
        
        # Technical indicators
        technical_keywords = ['code', 'program', 'debug', 'implement', 'develop', 'algorithm']
        
        # Complex query indicators
        complexity_indicators = ['detailed', 'comprehensive', 'thorough', 'in-depth', 'complete analysis']
        
        # Classify primary intent
        if any(keyword in query_lower for keyword in research_keywords):
            primary_intent = 'research'
            complexity = 'high' if len(user_query) > 100 else 'medium'
        elif any(keyword in query_lower for keyword in creative_keywords):
            primary_intent = 'creative'
            complexity = 'medium'
        elif any(keyword in query_lower for keyword in technical_keywords):
            primary_intent = 'technical'
            complexity = 'high'
        else:
            primary_intent = 'chat'
            complexity = 'low'
        
        # Adjust complexity based on query characteristics
        if any(indicator in query_lower for indicator in complexity_indicators):
            complexity = 'high'
        elif len(user_query) > 200:
            complexity = 'medium' if complexity == 'low' else 'high'
        
        # Determine confidence based on keyword matches
        confidence = 0.9 if any(keyword in query_lower for keyword in research_keywords + creative_keywords + technical_keywords) else 0.7
        
        return {
            'primary_intent': primary_intent,
            'secondary_intents': [],
            'confidence': confidence,
            'complexity': complexity,
            'query_length': len(user_query),
            'message_count': context.get('message_count', 0)
        }
    
    def _determine_rag_depth(self, intent_result: Dict[str, Any]) -> str:
        """
        Determine RAG depth ('SHALLOW' or 'DEEP') based on intent classification.
        
        This decision drives the conditional edge routing in RAG operations.
        """
        complexity = intent_result.get('complexity', 'medium')
        primary_intent = intent_result.get('primary_intent', 'chat')
        
        # Deep RAG for research, high complexity, or specific intents
        if (
            complexity == 'high' or 
            primary_intent in ['research', 'technical'] or
            intent_result.get('query_length', 0) > 150
        ):
            return 'DEEP'
        else:
            return 'SHALLOW'
    
    async def _assess_tool_requirements(
        self, 
        intent_result: Dict[str, Any], 
        conversation_ctx: ConversationCtx
    ) -> Dict[str, Any]:
        """
        Assess whether the intent requires tools and what kind.
        
        This drives the Dynamic Tool Agent workflow.
        """
        primary_intent = intent_result.get('primary_intent', 'chat')
        complexity = intent_result.get('complexity', 'medium')
        
        # Determine if tools are needed
        requires_tools = primary_intent in ['research', 'technical', 'creative'] or complexity == 'high'
        
        # Determine if external data is needed
        requires_external_data = primary_intent == 'research' or 'search' in intent_result.get('primary_intent', '')
        
        # Generate tool specification for dynamic tool creation
        tool_specification = ""
        if requires_tools:
            if primary_intent == 'research':
                tool_specification = "web search and content analysis tool"
            elif primary_intent == 'technical':
                tool_specification = "code analysis and generation tool"
            elif primary_intent == 'creative':
                tool_specification = "content generation and editing tool"
        
        return {
            'requires_tools': requires_tools,
            'requires_external_data': requires_external_data,
            'tool_specification': tool_specification
        }
    
    async def decide_search_depth(self, conversation_ctx: ConversationCtx) -> str:
        """
        Specialized method for RAG depth decision.
        
        This method is designed to be called by the `decide_search_depth` node
        in the workflow graph to set the `rag_depth_config` field.
        """
        intent_analysis = await self.analyze(conversation_ctx)
        return intent_analysis.rag_depth_recommendation