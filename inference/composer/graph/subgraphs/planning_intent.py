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

from typing import Annotated, List, Optional

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from models import (
    ChatResponse,
    IntentAnalysis,
    MessageContent,
    MessageContentType,
    MessageRole,
    Tool,
    WorkflowType,
    ComplexityLevel,
    TodoItem,
    Message,
)
from composer.graph import WorkflowExecutor, WorkflowState
from composer.agents.classifier_agent import ClassifierAgent
from utils.message_conversion import extract_text_from_message
from utils.logging import llmmllogger


logger = llmmllogger.bind(component="PlanningIntentSubgraph")


class PlanningIntentSubgraph:
    """
    Simplified intent analysis subgraph using LangGraph best practices.

    Direct approach:
    1. Analyze intent and complexity in one step
    2. Generate todos based on analysis results
    """

    def __init__(self, classifier_agent: ClassifierAgent):
        """Initialize intent analysis subgraph."""
        self.classifier_agent = classifier_agent
        self.graph: Optional[CompiledStateGraph] = None
        self._build_graph()

    def _build_graph(self) -> None:
        """Build simplified intent analysis subgraph using LangGraph patterns."""
        try:
            builder = StateGraph(WorkflowState)

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

    async def _analyze_intent_step(self, state: WorkflowState) -> WorkflowState:
        """Analyze intent and estimate complexity in one step."""
        logger.info("🔍 Intent: Analyzing messages for intent and complexity")
        static_tools = state.static_tools

        # Use classifier agent to analyze intent
        intent_analyses = await self.classifier_agent.analyze(
            messages=state.messages,
            available_static_tools=static_tools,
        )

        # Store intent analyses in database separately from message content
        await self._store_intent_analyses(intent_analyses, state)

        logger.info(
            f"🔍 Intent: Completed analysis with {len(intent_analyses)} intents"
        )

        # Cleanup classifier agent resources after analysis completion
        self.classifier_agent.cleanup()

        state.intent_classification = intent_analyses

        return state

    async def _store_intent_analyses(
        self,
        intent_analyses: List[IntentAnalysis],
        state: WorkflowState,
    ) -> None:
        """Store intent analyses in the database separately from message content."""
        try:
            from db import storage

            if not storage.initialized or not storage.analysis:
                logger.warning(
                    "Analysis storage not initialized, skipping intent analysis storage"
                )
                return

            messages = state.messages
            conversation_id = state.conversation_id

            if not messages or not conversation_id:
                logger.warning(
                    "Missing messages or conversation_id for intent analysis storage"
                )
                return

            # Find the latest user message to associate analyses with
            user_message_id = None
            for msg in reversed(messages):
                if hasattr(msg, "role") and getattr(msg, "role", "") == "user":
                    user_message_id = getattr(msg, "id", None)
                    break

            if not user_message_id:
                logger.warning(
                    "No user message found to associate intent analysis with"
                )
                return

            logger.debug(
                f"Found user message ID {user_message_id} for intent analysis storage"
            )

            # Verify the message exists in the database before storing analysis
            # Only check if storage is properly initialized to avoid errors
            if hasattr(storage, "message") and storage.message:
                try:
                    existing_message = await storage.get_service(
                        storage.message
                    ).get_message(user_message_id)
                    if not existing_message:
                        logger.error(
                            f"Message {user_message_id} not found in database - cannot store intent analysis"
                        )
                        return
                    logger.debug(
                        f"Verified message {user_message_id} exists in database"
                    )
                except Exception as e:
                    logger.error(
                        f"Error verifying message {user_message_id} exists: {e}"
                    )
                    # Don't return here - continue with storage attempt in case it's a transient issue
                    pass
            else:
                logger.warning(
                    "Message storage not available for verification - proceeding with intent analysis storage"
                )

            # Store each intent analysis
            for intent_analysis in intent_analyses:
                try:
                    # Set the message_id on the IntentAnalysis object before storing
                    intent_analysis.message_id = user_message_id

                    analysis_id = await storage.analysis.add_analysis(
                        message_id=user_message_id,
                        intent_analysis=intent_analysis,
                    )
                    if analysis_id:
                        logger.debug(f"Stored intent analysis with ID: {analysis_id}")
                    else:
                        logger.warning(
                            "Failed to store intent analysis - no ID returned"
                        )

                except Exception as e:
                    logger.error(f"Failed to store individual intent analysis: {e}")

        except Exception as e:
            logger.error(f"Failed to store intent analyses: {e}")

    async def _generate_todos_step(self, state: WorkflowState) -> WorkflowState:
        """Generate todos automatically based on intent analysis with proper typing."""
        logger.info("📝 Intent: Generating todos from intent analysis")

        if not state.intent_classification or not state.current_user_message:
            return state

        try:
            # Import storage here to avoid circular imports
            from db import storage

            if not storage.initialized or not storage.todo:
                logger.warning("Storage not initialized, skipping todo generation")
                return state

            # Generate todos based on intent analysis - simplified approach
            for intent in state.intent_classification:
                todo_item = await self._create_todo_for_intent(
                    intent,
                    extract_text_from_message(state.current_user_message),
                    state.user_id,
                    state.conversation_id,
                )
                if todo_item:
                    state.generated_todos.append(todo_item)

            logger.info(
                f"📝 Intent: Generated {len(state.generated_todos)} todos from intent analysis"
            )

            state.generated_todos

            return state

        except Exception as e:
            logger.error(f"Failed to generate todos from intent: {e}")
            return state

    async def _create_todo_for_intent(
        self,
        intent: IntentAnalysis,
        user_message: str,
        user_id: str,
        conversation_id: int,
    ) -> Optional[TodoItem]:
        """Create a single todo based on intent analysis using simplified logic."""
        try:
            from db import storage

            # Determine priority based on complexity and urgency indicators
            priority = "medium"  # default

            if intent.complexity_level == ComplexityLevel.COMPLEX:
                priority = "high"

            if any(
                word in user_message.lower()
                for word in ["urgent", "asap", "immediately", "quickly"]
            ):
                priority = "urgent"
            elif intent.complexity_level == ComplexityLevel.SIMPLE:
                priority = "low"

            # Create title based on workflow type
            workflow_action = {
                WorkflowType.RESEARCH: "Research",
                WorkflowType.ANALYSIS: "Analyze",
                WorkflowType.CREATIVE: "Create",
                WorkflowType.FOCUSED: "Execute",
                WorkflowType.PLANNING: "Plan",
            }.get(intent.workflow_type, "Work on")

            # Create description
            description = f"Complete {intent.workflow_type.value} {workflow_action} task: {user_message[:200]}..."
            if (
                hasattr(intent, "requires_custom_tools")
                and intent.requires_custom_tools
            ):
                description += " (Requires custom tools)"

            title = await self.classifier_agent.generate_title(
                [
                    Message(
                        role=MessageRole.USER,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT,
                                text=description,
                            )
                        ],
                    )
                ]
            )

            # Create TodoItem
            todo_item = TodoItem(
                user_id=user_id,
                conversation_id=conversation_id,
                description=description,
                status="not-started",
                priority=priority,
                title=title,
            )

            # Store in database and return the saved item
            saved_todo = await storage.get_service(storage.todo).add_todo(todo_item)
            return saved_todo

        except Exception as e:
            logger.error(f"Failed to create todo for intent: {e}")
            return None
