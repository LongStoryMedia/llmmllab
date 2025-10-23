"""
Planning Middleware Subgraph for Intent Analysis.

This subgraph implements sophisticated planning middleware patterns for intent analysis,
replacing the simple intent classifier node with a multi-step planning approach.

Key Features:
1. Multi-step intent analysis with planning
2. Context-aware decision making
3. Tool selection planning
4. Complexity estimation with planning middleware
"""

from typing import Dict, List, Any, Optional
from typing_extensions import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.graph.message import add_messages

from models import IntentAnalysis, WorkflowType, NodeMetadata, PipelinePriority
from composer.graph.state import WorkflowState
from composer.agents.classifier_agent import ClassifierAgent
from composer.utils.state import assemble_context_messages
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="PlanningIntentSubgraph")


class PlanningIntentState(TypedDict):
    """Minimal state for planning-based intent analysis."""
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str
    conversation_id: int
    static_tools: List[Any]
    planning_steps: List[str]
    complexity_score: int
    intent_analyses: List[IntentAnalysis]


class PlanningIntentSubgraph:
    """
    Planning middleware subgraph for sophisticated intent analysis.
    
    Implements multi-step planning approach:
    1. Initial context analysis
    2. Complexity estimation with planning
    3. Tool requirement planning
    4. Final intent classification
    """

    def __init__(self, classifier_agent: ClassifierAgent, pipeline_factory):
        """Initialize planning intent subgraph."""
        self.classifier_agent = classifier_agent
        self.pipeline_factory = pipeline_factory
        self.graph = None
        self._build_graph()

    def _build_graph(self) -> None:
        """Build the planning middleware subgraph."""
        try:
            builder = StateGraph(PlanningIntentState)

            # Planning middleware steps
            builder.add_node("context_analysis", self._context_analysis_step)
            builder.add_node("complexity_planning", self._complexity_planning_step)  
            builder.add_node("tool_planning", self._tool_planning_step)
            builder.add_node("intent_classification", self._intent_classification_step)

            # Planning middleware routing
            def should_continue_planning(state: PlanningIntentState):
                """Decide if more planning steps are needed."""
                planning_steps = state.get("planning_steps", [])
                complexity_score = state.get("complexity_score", 0)
                
                if len(planning_steps) < 2:
                    logger.info("🔀 Planning: Need more analysis steps")
                    return "complexity_planning"
                elif complexity_score > 7 and "tool_planning" not in planning_steps:
                    logger.info("🔀 Planning: High complexity, need tool planning")
                    return "tool_planning"
                else:
                    logger.info("🔀 Planning: Ready for intent classification")
                    return "intent_classification"

            # Build planning flow
            builder.add_edge(START, "context_analysis")
            
            builder.add_conditional_edges(
                "context_analysis",
                should_continue_planning,
                {
                    "complexity_planning": "complexity_planning",
                    "tool_planning": "tool_planning", 
                    "intent_classification": "intent_classification",
                }
            )
            
            builder.add_conditional_edges(
                "complexity_planning",
                should_continue_planning,
                {
                    "tool_planning": "tool_planning",
                    "intent_classification": "intent_classification",
                }
            )
            
            builder.add_edge("tool_planning", "intent_classification")
            builder.add_edge("intent_classification", END)

            self.graph = builder.compile()
            logger.info("Planning middleware subgraph built successfully")

        except Exception as e:
            logger.error(f"Failed to build planning subgraph: {e}")
            raise

    async def _context_analysis_step(self, state: PlanningIntentState) -> Dict[str, Any]:
        """Initial context analysis planning step."""
        logger.info("🔍 Planning: Context analysis step")
        
        messages = state.get("messages", [])
        planning_steps = state.get("planning_steps", [])
        
        # Analyze message context
        message_count = len(messages)
        has_recent_context = message_count > 1
        
        # Initial complexity estimation
        complexity_score = 3  # Base complexity
        if message_count > 5:
            complexity_score += 2
        if has_recent_context:
            complexity_score += 1
            
        planning_steps.append("context_analysis")
        
        return {
            "planning_steps": planning_steps,
            "complexity_score": complexity_score,
        }

    async def _complexity_planning_step(self, state: PlanningIntentState) -> Dict[str, Any]:
        """Planning middleware for complexity estimation."""
        logger.info("🔍 Planning: Complexity planning step")
        
        messages = state.get("messages", [])
        planning_steps = state.get("planning_steps", [])
        complexity_score = state.get("complexity_score", 3)
        
        # Analyze message content for complexity signals
        if messages:
            last_message = messages[-1]
            content = getattr(last_message, 'content', '').lower()
            
            # Technical complexity signals
            technical_keywords = ['algorithm', 'implementation', 'code', 'debug', 'error', 'api', 'database']
            research_keywords = ['research', 'analyze', 'compare', 'investigate', 'study', 'review']
            creative_keywords = ['create', 'design', 'generate', 'write', 'compose', 'draft']
            
            if any(kw in content for kw in technical_keywords):
                complexity_score += 3
            if any(kw in content for kw in research_keywords):
                complexity_score += 2  
            if any(kw in content for kw in creative_keywords):
                complexity_score += 1
                
            # Length-based complexity
            if len(content) > 200:
                complexity_score += 1
            if len(content) > 500:
                complexity_score += 2
                
        planning_steps.append("complexity_planning")
        
        return {
            "planning_steps": planning_steps,
            "complexity_score": min(complexity_score, 10),  # Cap at 10
        }

    async def _tool_planning_step(self, state: PlanningIntentState) -> Dict[str, Any]:
        """Planning middleware for tool requirement analysis."""
        logger.info("🔍 Planning: Tool planning step")
        
        messages = state.get("messages", [])
        planning_steps = state.get("planning_steps", [])
        static_tools = state.get("static_tools", [])
        
        # Analyze tool requirements based on content
        tool_requirements = []
        if messages:
            content = getattr(messages[-1], 'content', '').lower()
            
            if any(kw in content for kw in ['search', 'find', 'lookup', 'current', 'latest', 'news']):
                tool_requirements.append("web_search")
            if any(kw in content for kw in ['remember', 'recall', 'previous', 'before', 'history']):
                tool_requirements.append("memory_retrieval")
            if any(kw in content for kw in ['summarize', 'summary', 'brief', 'overview']):
                tool_requirements.append("summarization")
                
        planning_steps.append("tool_planning")
        
        logger.info(f"🔍 Planning: Identified tool requirements: {tool_requirements}")
        
        return {
            "planning_steps": planning_steps,
            "tool_requirements": tool_requirements,
        }

    async def _intent_classification_step(self, state: PlanningIntentState) -> Dict[str, Any]:
        """Final intent classification with planning context."""
        logger.info("🔍 Planning: Intent classification step")
        
        messages = state.get("messages", [])
        static_tools = state.get("static_tools", [])
        complexity_score = state.get("complexity_score", 3)
        planning_steps = state.get("planning_steps", [])
        
        # Convert to LangChain messages for classifier
        langchain_messages = []
        for msg in messages:
            if isinstance(msg, (HumanMessage, AIMessage)):
                langchain_messages.append(msg)
                
        # Use classifier agent with planning context
        intent_analyses = await self.classifier_agent.analyze(
            messages=langchain_messages,
            available_static_tools=static_tools,
        )
        
        # Enhance intent analyses with planning context
        for intent in intent_analyses:
            if hasattr(intent, 'complexity_estimate'):
                intent.complexity_estimate = max(intent.complexity_estimate, complexity_score)
        
        planning_steps.append("intent_classification")
        
        logger.info(f"🔍 Planning: Completed with {len(intent_analyses)} intent analyses")
        
        return {
            "planning_steps": planning_steps,
            "intent_analyses": intent_analyses,
        }

    def transform_to_planning_state(self, main_state: WorkflowState) -> PlanningIntentState:
        """Transform main WorkflowState to PlanningIntentState."""
        messages = main_state.messages[-5:] if main_state.messages else []
        
        # Convert to LangChain core messages
        langchain_messages = []
        for msg in messages:
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                if msg.type == 'human':
                    langchain_messages.append(HumanMessage(content=msg.content))
                elif msg.type == 'ai':
                    langchain_messages.append(AIMessage(content=msg.content))
                    
        return {
            "messages": langchain_messages,
            "user_id": getattr(main_state, "user_id", ""),
            "conversation_id": getattr(main_state, "conversation_id", 0),
            "static_tools": getattr(main_state, "static_tools", []),
            "planning_steps": [],
            "complexity_score": 3,
            "intent_analyses": [],
        }

    def transform_to_main_state(self, planning_result: Dict[str, Any], main_state: WorkflowState) -> Dict[str, Any]:
        """Transform planning results back to main WorkflowState updates."""
        updates = {}
        
        if planning_result.get("intent_analyses"):
            # Extend the intent classification list
            current_analyses = getattr(main_state, "intent_classification", [])
            updates["intent_classification"] = current_analyses + planning_result["intent_analyses"]
            
        return updates

    async def execute(self, main_state: WorkflowState) -> Command:
        """Execute the planning middleware subgraph."""
        try:
            if not self.graph:
                logger.error("Planning subgraph not initialized")
                return Command(update={})

            # Transform to planning state
            planning_state = self.transform_to_planning_state(main_state)

            # Execute planning subgraph
            result = await self.graph.ainvoke(
                planning_state,
                config={"recursion_limit": 10}
            )

            # Transform results back
            updates = self.transform_to_main_state(result, main_state)

            logger.info(f"🔍 Planning: Subgraph completed with {len(updates)} updates")
            return Command(update=updates)

        except Exception as e:
            logger.error(f"Planning subgraph execution failed: {e}", exc_info=True)
            return Command(update={})


# Global instance
planning_intent_subgraph = None

def get_planning_intent_subgraph():
    """Get or create planning intent subgraph instance."""
    global planning_intent_subgraph
    if planning_intent_subgraph is None:
        # Import here to avoid circular imports
        from runner.pipeline_factory import pipeline_factory
        from models.default_model_profiles import DEFAULT_ANALYSIS_PROFILE
        
        # Create classifier agent
        classifier_agent = ClassifierAgent(
            pipeline_factory=pipeline_factory,
            profile=DEFAULT_ANALYSIS_PROFILE,
            node_metadata=NodeMetadata(
                node_name="PlanningClassifierAgent",
                node_id="planning_classifier",
                node_type="agent",
                user_id="system",
                conversation_id=0,
            ),
        )
        
        planning_intent_subgraph = PlanningIntentSubgraph(classifier_agent, pipeline_factory)
    
    return planning_intent_subgraph