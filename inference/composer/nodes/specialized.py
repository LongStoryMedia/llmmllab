"""
Specialized LangGraph nodes for advanced workflow operations.
Implements TitleGenerationNode, IntentClassifierAgent, EngineeringAgentNode per Phase 2 requirements.
"""

import asyncio
from typing import Dict, Any, Optional, List

from models import (
    LangChainMessage, 
    IntentAnalysis, 
    AvailableTool, 
    ModelProfileType,
    DynamicToolSpec
)

# Lazy imports to avoid circular dependencies
# from db import storage - imported when needed
# from utils.model_profile_utils import get_model_profile_for_task - imported when needed  
# from utils.grammar_generator import GrammarGenerator - imported when needed

from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError, ToolGenerationError
from composer.tools.registry import ToolRegistry


class TitleGenerationNode:
    """
    Generates a conversation title if none exists.
    
    Uses grammar-constrained LLM to generate concise, descriptive titles
    based on conversation content.
    """

    def __init__(self, pipeline_factory):
        """
        Initialize title generation node.
        
        Args:
            pipeline_factory: Factory for creating structured pipelines
        """
        self.pipeline_factory = pipeline_factory
        self.logger = composer_logger.bind(component="TitleGenerationNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Generate conversation title if needed.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with title
        """
        try:
            # Skip if title already exists
            if hasattr(state, 'conversation_title') and state.conversation_title:
                return state

            # Need at least 2 messages (user + assistant) to generate meaningful title
            if len(state.messages) < 2:
                return state

            user_id = getattr(state, 'user_id', None)
            if not user_id:
                return state

            # Get model profile for analysis tasks
            uc = await storage.get_service(storage.user_config).get_user_config(user_id)
            model_profile = get_model_profile_for_task(
                uc.model_profiles, ModelProfileType.Analysis, user_id
            )

            self.logger.info(
                "Generating conversation title",
                user_id=user_id,
                message_count=len(state.messages)
            )

            # Create title generation pipeline with grammar constraints
            title_pipeline = await self.pipeline_factory.create_structured_pipeline(
                prompt_template=self._get_title_prompt(),
                output_schema=str,  # Simple string output
                enable_fallback=True
            )

            # Format conversation context
            conversation_context = self._format_conversation_context(state.messages)

            # Generate title
            title = await title_pipeline.execute({
                "conversation": conversation_context
            })

            # Update state with generated title
            state.conversation_title = title.strip()
            
            self.logger.info(
                "Title generated successfully",
                user_id=user_id,
                title=state.conversation_title
            )

            return state

        except Exception as e:
            self.logger.error(
                "Title generation failed",
                user_id=getattr(state, 'user_id', 'unknown'),
                error=str(e)
            )
            
            # Continue without title on error
            state.conversation_title = "Untitled Conversation"
            return state

    def _get_title_prompt(self) -> str:
        """Get the title generation prompt template."""
        return """Generate a concise, descriptive title for this conversation.

Conversation:
{conversation}

Requirements:
- Maximum 8 words
- Capture the main topic or question
- Use clear, simple language
- No quotes or special characters

Title:"""

    def _format_conversation_context(self, messages: List[LangChainMessage]) -> str:
        """Format messages for title generation context."""
        context_lines = []
        
        for i, message in enumerate(messages[:6]):  # Use first 6 messages
            role = "User" if message.role == "user" else "Assistant"
            content = message.content[:200]  # Truncate long messages
            context_lines.append(f"{role}: {content}")
        
        return "\n".join(context_lines)


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
        self.logger = composer_logger.bind(component="IntentClassifierNode")
        
        # Generate grammar for intent analysis at initialization
        self.intent_grammar = GrammarGenerator.from_pydantic_model(IntentAnalysis)

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
                raise NodeExecutionError("User ID required for intent classification")

            # Get latest user message
            user_messages = [msg for msg in state.messages if msg.role == "user"]
            if not user_messages:
                return state

            latest_query = user_messages[-1].content

            # Get model profile for analysis tasks
            uc = await storage.get_service(storage.user_config).get_user_config(user_id)
            model_profile = get_model_profile_for_task(
                uc.model_profiles, ModelProfileType.Analysis, user_id
            )

            self.logger.info(
                "Analyzing user intent",
                user_id=user_id,
                query_length=len(latest_query)
            )

            # Create grammar-constrained pipeline for intent analysis
            structured_pipeline = await self.pipeline_factory.create_structured_pipeline(
                prompt_template=self._get_intent_prompt(),
                output_schema=IntentAnalysis,
                grammar=self.intent_grammar,
                enable_fallback=True
            )

            # Execute intent analysis with structured output
            intent_analysis = await structured_pipeline.execute({
                "user_query": latest_query,
                "conversation_context": self._format_context(state.messages)
            })

            # Update state with validated structured output
            state.intent_classification = intent_analysis
            state.rag_depth_config = intent_analysis.search_complexity.value

            self.logger.info(
                "Intent classification completed",
                user_id=user_id,
                intent_category=intent_analysis.intent_category.value,
                search_complexity=intent_analysis.search_complexity.value,
                needs_tools=intent_analysis.requires_tools
            )

            return state

        except Exception as e:
            self.logger.error(
                "Intent classification failed",
                user_id=getattr(state, 'user_id', 'unknown'),
                error=str(e)
            )
            
            # Create fallback intent analysis
            fallback_intent = IntentAnalysis(
                intent_category="general_chat",
                search_complexity="SHALLOW",
                requires_tools=False,
                confidence_score=0.1
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
2. Required search depth (SHALLOW for simple questions, DEEP for research)
3. Tool requirements (does this need external tools, calculations, etc.)
4. Complexity level and resource requirements
5. Confidence in classification (0.0 to 1.0)

Consider:
- Simple questions = SHALLOW search
- Research topics, analysis, current events = DEEP search  
- Calculations, data processing = tools required
- Creative tasks may need specialized tools

Respond with valid JSON matching the IntentAnalysis schema.
All fields are required and must conform to the specified enums and constraints."""

    def _format_context(self, messages: List[LangChainMessage]) -> str:
        """Format conversation context for intent analysis."""
        if len(messages) <= 1:
            return "No prior context"
        
        context_lines = []
        for msg in messages[-5:]:  # Last 5 messages for context
            role = msg.role.title()
            content = msg.content[:150]  # Truncate for brevity
            context_lines.append(f"{role}: {content}")
        
        return "\n".join(context_lines)


class EngineeringAgentNode:
    """
    Engineering Agent for dynamic tool generation with grammar-constrained structured output.
    
    Generates, retrieves, or composes tools based on intent analysis and user requirements.
    Implements the three-tier decision process: Use Existing -> Modify/Compose -> Create New.
    """

    def __init__(self, pipeline_factory):
        """
        Initialize engineering agent node.
        
        Args:
            pipeline_factory: Factory for creating structured pipelines
        """
        self.pipeline_factory = pipeline_factory
        self.tool_registry = ToolRegistry()
        self.logger = composer_logger.bind(component="EngineeringAgentNode")
        
        # Generate grammar for dynamic tool specification
        self.tool_spec_grammar = GrammarGenerator.from_pydantic_model(DynamicToolSpec)

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Generate or retrieve dynamic tools based on intent analysis.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with required tools
        """
        try:
            # Skip if no intent classification available
            if not state.intent_classification:
                return state

            intent = state.intent_classification
            user_id = getattr(state, 'user_id', None)
            
            if not user_id:
                raise NodeExecutionError("User ID required for tool orchestration")

            self.logger.info(
                "Engineering agent processing tool requirements",
                user_id=user_id,
                requires_tools=intent.requires_tools,
                intent_category=intent.intent_category.value
            )

            # Get standard tools based on intent
            tools = await self.tool_registry.get_tools_for_context(intent, user_id)
            
            # Check if dynamic tool generation is needed
            if intent.requires_tools and intent.requires_specialized_tools:
                dynamic_tool = await self._generate_or_retrieve_dynamic_tool(
                    state, intent, user_id
                )
                if dynamic_tool:
                    tools.append(dynamic_tool)

            # Update state with available tools
            available_tools = [
                AvailableTool(
                    name=tool.name,
                    description=tool.description,
                    parameters=getattr(tool, 'parameters', {}),
                    tool_type="static" if hasattr(tool, '_static') else "dynamic"
                )
                for tool in tools
            ]
            
            state.required_tools = available_tools

            self.logger.info(
                "Tool orchestration completed",
                user_id=user_id,
                tool_count=len(tools),
                tool_names=[tool.name for tool in tools]
            )

            return state

        except Exception as e:
            self.logger.error(
                "Engineering agent execution failed",
                user_id=getattr(state, 'user_id', 'unknown'),
                error=str(e)
            )
            
            # Continue with empty tools on error
            state.required_tools = []
            return state

    async def _generate_or_retrieve_dynamic_tool(
        self, 
        state: WorkflowState, 
        intent: IntentAnalysis, 
        user_id: str
    ) -> Optional[Any]:
        """
        Generate or retrieve dynamic tool based on requirements.
        
        Args:
            state: Current workflow state
            intent: Intent analysis results
            user_id: User identifier
            
        Returns:
            Dynamic tool instance or None
        """
        try:
            # Extract tool requirements from user query
            user_query = state.messages[-1].content if state.messages else ""
            
            # Create tool specification using Engineering Agent
            tool_spec = await self._generate_tool_spec_with_engineering_agent(
                user_query, intent, user_id
            )
            
            if not tool_spec:
                return None

            # Use ToolRegistry's three-tier decision process
            dynamic_tool = await self.tool_registry.find_or_create_tool(
                tool_spec, user_id
            )
            
            return dynamic_tool

        except Exception as e:
            self.logger.error(
                "Dynamic tool generation failed",
                user_id=user_id,
                error=str(e)
            )
            return None

    async def _generate_tool_spec_with_engineering_agent(
        self, 
        user_query: str, 
        intent: IntentAnalysis, 
        user_id: str
    ) -> Optional[DynamicToolSpec]:
        """
        Generate structured tool specification using Engineering Agent with grammar constraints.
        
        Args:
            user_query: User's original query
            intent: Intent analysis results
            user_id: User identifier
            
        Returns:
            Structured tool specification
        """
        try:
            # Get model profile for tool generation
            uc = await storage.get_service(storage.user_config).get_user_config(user_id)
            model_profile = get_model_profile_for_task(
                uc.model_profiles, ModelProfileType.Primary, user_id
            )

            # Create grammar-constrained Engineering Agent pipeline
            engineering_pipeline = await self.pipeline_factory.create_structured_pipeline(
                prompt_template=self._get_engineering_agent_prompt(),
                output_schema=DynamicToolSpec,
                grammar=self.tool_spec_grammar,
                enable_fallback=True
            )

            # Generate structured tool specification
            tool_spec = await engineering_pipeline.execute({
                "user_query": user_query,
                "intent_category": intent.intent_category.value,
                "requires_external_data": intent.requires_external_data,
                "complexity": intent.complexity_level.value
            })

            return tool_spec

        except Exception as e:
            self.logger.error(
                "Tool spec generation failed",
                user_id=user_id,
                error=str(e)
            )
            return None

    def _get_engineering_agent_prompt(self) -> str:
        """Get the Engineering Agent prompt template."""
        return """As an Engineering Agent, analyze the user's request and generate a tool specification.

User Query: {user_query}
Intent Category: {intent_category}
Requires External Data: {requires_external_data}
Complexity Level: {complexity}

Create a tool specification that:
1. Defines the tool's purpose and functionality
2. Specifies input parameters and their types
3. Describes the expected output format
4. Includes implementation approach (API calls, calculations, etc.)
5. Considers security and validation requirements

Tool Requirements:
- Must be composable using LangChain LCEL patterns
- Should have clear input/output schema
- Must include proper error handling
- Should be efficient and focused on single responsibility

Generate a structured tool specification that can be compiled into a working tool.
Respond with valid JSON matching the DynamicToolSpec schema."""