"""
Planning Middleware Subgraph for Intent Analysis.

This subgraph implements sophisticated planning middleware patterns for intent analysis,
replacing the simple intent classifier node with a multi-step planning approach.

Key Features:
1. Multi-step intent analysis with planning
2. Context-aware decision making
3. Tool selection planning
4. Complexity estimation with planning middleware

Memory & Persistence:
- Automatic checkpoint inheritance from parent graph (per LangGraph docs)
- Planning steps, complexity scores, and todos persist across turns
- No manual checkpoint setup required - LangGraph handles propagation
- State restoration automatic when parent workflow resumes
"""

from typing import Dict, List, Any, Optional
from typing_extensions import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from langgraph.graph.message import add_messages

from models import (
    IntentAnalysis,
    WorkflowType,
    NodeMetadata,
    ComplexityLevel,
    TodoItem,
    Tool,
)
from composer.graph.state import WorkflowState
from composer.agents.classifier_agent import ClassifierAgent
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="PlanningIntentSubgraph")


class PlanningIntentState(TypedDict):
    """Simplified state for intent analysis with stronger typing."""

    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str
    conversation_id: int
    static_tools: List[Tool]
    intent_analyses: List[IntentAnalysis]
    generated_todos: List[TodoItem]
    complexity_score: int


class PlanningIntentSubgraph:
    """
    Simplified intent analysis subgraph using LangGraph best practices.

    Direct approach:
    1. Analyze intent and complexity in one step
    2. Generate todos based on analysis results
    """

    def __init__(self, classifier_agent: ClassifierAgent, pipeline_factory):
        """Initialize intent analysis subgraph."""
        self.classifier_agent = classifier_agent
        self.pipeline_factory = pipeline_factory
        self.graph: Optional[CompiledStateGraph] = None
        self._build_graph()

    def _build_graph(self) -> None:
        """Build simplified intent analysis subgraph using LangGraph patterns."""
        try:
            builder = StateGraph(PlanningIntentState)

            # Simplified two-step approach
            builder.add_node("analyze_intent", self._analyze_intent_step)
            builder.add_node("generate_todos", self._generate_todos_step)

            # Simple linear flow
            builder.add_edge(START, "analyze_intent")
            builder.add_edge("analyze_intent", "generate_todos")
            builder.add_edge("generate_todos", END)

            self.graph = builder.compile()
            logger.info("Intent analysis subgraph built successfully")

        except Exception as e:
            logger.error(f"Failed to build intent analysis subgraph: {e}")
            raise

    async def _analyze_intent_step(self, state: PlanningIntentState) -> Dict[str, Any]:
        """Analyze intent and estimate complexity in one step."""
        logger.info("🔍 Intent: Analyzing messages for intent and complexity")

        messages = state.get("messages", [])
        static_tools = state.get("static_tools", [])

        # Convert to LangChain messages for classifier
        langchain_messages = []
        for msg in messages:
            if isinstance(msg, (HumanMessage, AIMessage)):
                langchain_messages.append(msg)

        # Use classifier agent to analyze intent (no streaming output to prevent leakage)
        intent_analyses = await self.classifier_agent.analyze(
            messages=langchain_messages,
            available_static_tools=static_tools,
        )
        
        # Store intent analyses in database separately from message content
        await self._store_intent_analyses(intent_analyses, state)

        # Calculate complexity score based on analysis results
        complexity_score = self._calculate_complexity_score(messages, intent_analyses)

        logger.info(
            f"🔍 Intent: Completed analysis with {len(intent_analyses)} intents, complexity: {complexity_score}"
        )

        return {
            "intent_analyses": intent_analyses,
            "complexity_score": complexity_score,
        }

    def _calculate_complexity_score(
        self, messages: List[BaseMessage], intent_analyses: List[IntentAnalysis]
    ) -> int:
        """Calculate complexity score based on messages and intent analysis."""
        complexity_score = 3  # Base complexity

        # Message-based complexity
        complexity_score += min(len(messages), 5)  # More messages = more complexity

        if messages:
            last_message = messages[-1]
            content = getattr(last_message, "content", "").lower()

            # Keyword-based complexity indicators
            technical_keywords = [
                "algorithm",
                "implementation",
                "code",
                "debug",
                "error",
                "api",
                "database",
            ]
            research_keywords = [
                "research",
                "analyze",
                "compare",
                "investigate",
                "study",
                "review",
            ]
            creative_keywords = [
                "create",
                "design",
                "generate",
                "write",
                "compose",
                "draft",
            ]

            if any(kw in content for kw in technical_keywords):
                complexity_score += 3
            elif any(kw in content for kw in research_keywords):
                complexity_score += 2
            elif any(kw in content for kw in creative_keywords):
                complexity_score += 1

            # Length-based complexity
            if len(content) > 500:
                complexity_score += 2
            elif len(content) > 200:
                complexity_score += 1

        # Intent analysis-based complexity
        for intent in intent_analyses:
            if hasattr(intent, "complexity_level"):
                if intent.complexity_level == ComplexityLevel.COMPLEX:
                    complexity_score += 2
                elif intent.complexity_level == ComplexityLevel.MODERATE:
                    complexity_score += 1

            if (
                hasattr(intent, "requires_custom_tools")
                and intent.requires_custom_tools
            ):
                complexity_score += 2

        return min(complexity_score, 10)  # Cap at 10

    async def _store_intent_analyses(
        self, intent_analyses: List[IntentAnalysis], state: PlanningIntentState
    ) -> None:
        """Store intent analyses in the database separately from message content."""
        try:
            from db import storage
            
            if not storage.initialized or not storage.analysis:
                logger.warning("Analysis storage not initialized, skipping intent analysis storage")
                return
            
            messages = state.get("messages", [])
            conversation_id = state.get("conversation_id")
            
            if not messages or not conversation_id:
                logger.warning("Missing messages or conversation_id for intent analysis storage")
                return
                
            # Find the latest user message to associate analyses with
            user_message_id = None
            for msg in reversed(messages):
                if hasattr(msg, "role") and getattr(msg, "role", "") == "user":
                    user_message_id = getattr(msg, "id", None)
                    break
                    
            if not user_message_id:
                logger.warning("No user message found to associate intent analysis with")
                return
            
            # Store each intent analysis
            for intent_analysis in intent_analyses:
                try:
                    analysis_id = await storage.analysis.add_analysis(
                        message_id=user_message_id,
                        intent_analysis=intent_analysis
                    )
                    if analysis_id:
                        logger.debug(f"Stored intent analysis with ID: {analysis_id}")
                    else:
                        logger.warning("Failed to store intent analysis - no ID returned")
                        
                except Exception as e:
                    logger.error(f"Failed to store individual intent analysis: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to store intent analyses: {e}")

    async def _generate_todos_step(self, state: PlanningIntentState) -> Dict[str, Any]:
        """Generate todos based on intent analysis."""
        logger.info("� Intent: Generating todos from intent analysis")

        intent_analyses = state.get("intent_analyses", [])
        messages = state.get("messages", [])
        user_id = state.get("user_id")
        conversation_id = state.get("conversation_id")
        complexity_score = state.get("complexity_score", 3)

        generated_todos = await self._generate_todos_from_intent(
            intent_analyses, messages, user_id, conversation_id, complexity_score
        )

        logger.info(f"� Intent: Generated {len(generated_todos)} todos")

        return {
            "generated_todos": generated_todos,
        }

    async def _generate_todos_from_intent(
        self,
        intent_analyses: List[IntentAnalysis],
        messages: List[BaseMessage],
        user_id: str,
        conversation_id: int,
        complexity_score: int,
    ) -> List[TodoItem]:
        """Generate todos automatically based on intent analysis with proper typing."""
        logger.info("📝 Intent: Generating todos from intent analysis")

        if not intent_analyses or not user_id:
            return []

        generated_todos: List[TodoItem] = []

        try:
            # Import storage here to avoid circular imports
            from db import storage

            if not storage.initialized or not storage.todo:
                logger.warning("Storage not initialized, skipping todo generation")
                return []

            # Get the latest user message for context
            user_message = ""
            if messages:
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        user_message = getattr(msg, "content", "")
                        break

            # Generate todos based on intent analysis - simplified approach
            for intent in intent_analyses:
                todo_item = await self._create_todo_for_intent(
                    intent, user_message, user_id, conversation_id, complexity_score
                )
                if todo_item:
                    generated_todos.append(todo_item)

            logger.info(
                f"📝 Intent: Generated {len(generated_todos)} todos from intent analysis"
            )
            return generated_todos

        except Exception as e:
            logger.error(f"Failed to generate todos from intent: {e}")
            return []

    async def _create_todo_for_intent(
        self,
        intent: IntentAnalysis,
        user_message: str,
        user_id: str,
        conversation_id: int,
        complexity_score: int,
    ) -> Optional[TodoItem]:
        """Create a single todo based on intent analysis using simplified logic."""
        try:
            from db import storage

            # Determine priority based on complexity and urgency indicators
            priority = "medium"  # default

            if complexity_score >= 8 or (
                hasattr(intent, "complexity_level")
                and intent.complexity_level == ComplexityLevel.COMPLEX
            ):
                priority = "high"
            elif any(
                word in user_message.lower()
                for word in ["urgent", "asap", "immediately", "quickly"]
            ):
                priority = "urgent"
            elif complexity_score <= 3 or (
                hasattr(intent, "complexity_level")
                and intent.complexity_level == ComplexityLevel.SIMPLE
            ):
                priority = "low"

            # Create title based on workflow type
            workflow_action = {
                WorkflowType.RESEARCH: "Research",
                WorkflowType.ANALYSIS: "Analyze",
                WorkflowType.CREATIVE: "Create",
                WorkflowType.FOCUSED: "Execute",
                WorkflowType.PLANNING: "Plan",
            }.get(intent.workflow_type, "Work on")

            topic = self._extract_topic(user_message)
            title = f"{workflow_action}: {topic}"

            # Create description
            description = (
                f"Complete {intent.workflow_type.value} task: {user_message[:200]}..."
            )
            if (
                hasattr(intent, "requires_custom_tools")
                and intent.requires_custom_tools
            ):
                description += " (Requires custom tools)"

            # Create TodoItem
            todo_item = TodoItem(
                user_id=user_id,
                conversation_id=conversation_id,
                title=title,
                description=description,
                status="not-started",
                priority=priority,
            )

            # Store in database and return the saved item
            saved_todo = await storage.get_service(storage.todo).add_todo(todo_item)
            return saved_todo

        except Exception as e:
            logger.error(f"Failed to create todo for intent: {e}")
            return None

    def _extract_topic(self, message: str) -> str:
        """Extract a concise topic from the user message for todo titles."""
        # Simple topic extraction - take first meaningful part of message
        words = message.split()
        if len(words) <= 5:
            return message

        # Look for key topic indicators
        topic_words = []
        skip_words = {
            "please",
            "can",
            "you",
            "help",
            "me",
            "with",
            "i",
            "need",
            "want",
            "would",
            "like",
        }

        for word in words[:10]:  # Look at first 10 words
            clean_word = word.lower().strip(".,!?")
            if clean_word not in skip_words and len(clean_word) > 2:
                topic_words.append(word)
                if len(topic_words) >= 3:
                    break

        if topic_words:
            return " ".join(topic_words)
        else:
            return " ".join(words[:3])

    def transform_to_planning_state(
        self, main_state: WorkflowState
    ) -> PlanningIntentState:
        """Transform main WorkflowState to PlanningIntentState with proper typing."""
        messages = main_state.messages[-5:] if main_state.messages else []

        # Convert to LangChain core messages
        langchain_messages: List[BaseMessage] = []
        for msg in messages:
            if hasattr(msg, "type") and hasattr(msg, "content"):
                if msg.type == "human":
                    langchain_messages.append(HumanMessage(content=msg.content))
                elif msg.type == "ai":
                    langchain_messages.append(AIMessage(content=msg.content))

        # Get static tools with proper typing
        static_tools: List[Tool] = getattr(main_state, "static_tools", [])

        # Get existing todos with proper typing
        existing_todos: List[TodoItem] = getattr(main_state, "generated_todos", [])

        return {
            "messages": langchain_messages,
            "user_id": getattr(main_state, "user_id", ""),
            "conversation_id": getattr(main_state, "conversation_id", 0),
            "static_tools": static_tools,
            "intent_analyses": [],
            "generated_todos": existing_todos,
            "complexity_score": getattr(main_state, "complexity_score", 3),
        }

    def transform_to_main_state(
        self, planning_result: Dict[str, Any], main_state: WorkflowState
    ) -> Dict[str, Any]:
        """Transform planning results back to main WorkflowState updates with proper typing."""
        updates: Dict[str, Any] = {}

        if planning_result.get("intent_analyses"):
            # Extend the intent classification list
            current_analyses: List[IntentAnalysis] = getattr(
                main_state, "intent_classification", []
            )
            updates["intent_classification"] = (
                current_analyses + planning_result["intent_analyses"]
            )

        # Include generated todos in the main state with proper typing
        if planning_result.get("generated_todos"):
            generated_todos: List[TodoItem] = planning_result["generated_todos"]
            updates["generated_todos"] = generated_todos

        # Include complexity score
        if planning_result.get("complexity_score"):
            updates["complexity_score"] = planning_result["complexity_score"]

        return updates

    async def execute(self, main_state: WorkflowState) -> Command:
        """Execute the simplified intent analysis subgraph."""
        try:
            if not self.graph:
                logger.error("Intent analysis subgraph not initialized")
                return Command(update={})

            # Transform to planning state
            planning_state = self.transform_to_planning_state(main_state)

            # Execute simplified subgraph with reasonable recursion limit
            result = await self.graph.ainvoke(
                planning_state, config={"recursion_limit": 5}
            )

            # Transform results back
            updates = self.transform_to_main_state(result, main_state)

            logger.info(f"🔍 Intent: Analysis completed with {len(updates)} updates")
            return Command(update=updates)

        except Exception as e:
            logger.error(
                f"Intent analysis subgraph execution failed: {e}", exc_info=True
            )
            return Command(update={})


# Global instance
planning_intent_subgraph = None


def get_planning_intent_subgraph():
    """Get or create planning intent subgraph instance."""
    global planning_intent_subgraph
    if planning_intent_subgraph is None:
        # Import here to avoid circular imports
        from runner.pipeline_factory import pipeline_factory
        from models.default_model_profiles import DEFAULT_ANALYSIS_PROFILE, DEFAULT_PRIMARY_MODEL
        from models.model_profile_config import ModelProfile, ModelParameters
        import uuid

        # TEMP FIX: Create modified analysis profile using multimodal model instead of qwen3-4b
        # This avoids grammar constraints that crash with multimodal content
        temp_analysis_profile = ModelProfile(
            id=uuid.uuid4(),  # Temporary ID
            user_id="system",
            name="Analysis (Multimodal Compatible)",
            type="analysis",
            description="Profile for analysis using multimodal model to avoid grammar constraints",
            model_name="qwen3-vl-32b-thinking-abliterated",  # Use multimodal model instead of qwen3-4b
            parameters=DEFAULT_ANALYSIS_PROFILE.parameters  # Keep same parameters
        )

        # Create classifier agent
        classifier_agent = ClassifierAgent(
            pipeline_factory=pipeline_factory,
            profile=temp_analysis_profile,  # Using modified profile with qwen3vl instead of qwen3-4b
            node_metadata=NodeMetadata(
                node_name="PlanningClassifierAgent",
                node_id="planning_classifier",
                node_type="agent",
                user_id="system",
                conversation_id=0,
            ),
        )

        planning_intent_subgraph = PlanningIntentSubgraph(
            classifier_agent, pipeline_factory
        )

    return planning_intent_subgraph
