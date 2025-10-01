"""
Engineering agent node for dynamic tool generation and orchestration.
Generates, retrieves, or composes tools based on intent analysis and user requirements.
"""

from typing import List, Optional, Any

from models.intent_analysis import IntentAnalysis
from models.available_tool import AvailableTool
from models.dynamic_tool import DynamicTool
from models.model_profile_type import ModelProfileType
from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError
from composer.tools.registry import ToolRegistry

# Lazy imports to avoid circular dependencies
from db import storage  # pylint: disable=import-outside-toplevel
from utils.model_profile import get_model_profile_for_task  # pylint: disable=import-outside-toplevel
from utils.grammar_generator import get_grammar_for_model  # pylint: disable=import-outside-toplevel


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
        self.logger = composer_logger.logger

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
                raise NodeExecutionError("engineering_agent", Exception("User ID required for tool orchestration"))

            self.logger.info(
                "Engineering agent processing tool requirements",
                extra={
                    "user_id": user_id,
                    "primary_intent": intent.primary_intent,
                    "required_capabilities": [str(cap) for cap in intent.required_capabilities]
                }
            )

            # Get standard tools based on intent
            tools = await self.tool_registry.get_tools_for_context(intent, user_id)
            
            # Check if dynamic tool generation is needed based on capabilities
            needs_dynamic_tools = any(
                'DYNAMIC' in str(cap) or 'SPECIALIZED' in str(cap) 
                for cap in intent.required_capabilities
            )
            
            if needs_dynamic_tools:
                dynamic_tool = await self._generate_or_retrieve_dynamic_tool(
                    state, intent, user_id
                )
                if dynamic_tool:
                    tools.append(dynamic_tool)

            # Update state with available tools
            available_tools = [
                AvailableTool(
                    name=getattr(tool, 'name', 'unnamed_tool'),
                    description=getattr(tool, 'description', 'No description available'),
                    parameters=getattr(tool, 'parameters', {}),
                    tool_type="static" if hasattr(tool, '_static') else "dynamic"
                )
                for tool in tools if tool is not None
            ]
            
            state.required_tools = available_tools

            self.logger.info(
                "Tool orchestration completed",
                extra={
                    "user_id": user_id,
                    "tool_count": len(tools),
                    "tool_names": [tool.name for tool in available_tools]
                }
            )

            return state

        except Exception as e:
            self.logger.error(
                "Engineering agent execution failed",
                extra={
                    "user_id": getattr(state, 'user_id', 'unknown'),
                    "error": str(e)
                }
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
            user_query = getattr(state.messages[-1], 'content', '') if state.messages else ""
            
            # Create tool specification using Engineering Agent
            tool_spec = await self._generate_tool_spec_with_engineering_agent(
                user_query, intent, user_id
            )
            
            if not tool_spec:
                return None

            # Use ToolRegistry's dynamic tool generation method
            dynamic_tool = await self.tool_registry._generate_or_retrieve_dynamic_tool(
                tool_spec, user_id
            )
            
            return dynamic_tool

        except Exception as e:
            self.logger.error(
                "Dynamic tool generation failed",
                extra={
                    "user_id": user_id,
                    "error": str(e)
                }
            )
            return None

    async def _generate_tool_spec_with_engineering_agent(
        self, 
        user_query: str, 
        intent: IntentAnalysis, 
        user_id: str
    ) -> Optional[DynamicTool]:
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
                output_schema=DynamicTool,
                grammar=get_grammar_for_model(DynamicTool),
                enable_fallback=True
            )

            # Generate structured tool specification
            tool_spec = await engineering_pipeline.execute({
                "user_query": user_query,
                "primary_intent": intent.primary_intent,
                "required_capabilities": [str(cap) for cap in intent.required_capabilities],
                "complexity": intent.complexity_level.value
            })

            return tool_spec

        except Exception as e:
            self.logger.error(
                "Tool spec generation failed",
                extra={
                    "user_id": user_id,
                    "error": str(e)
                }
            )
            return None

    def _get_engineering_agent_prompt(self) -> str:
        """Get the Engineering Agent prompt template."""
        return """As an Engineering Agent, analyze the user's request and generate a tool specification.

User Query: {user_query}
Primary Intent: {primary_intent}
Required Capabilities: {required_capabilities}
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
Respond with valid JSON matching the DynamicTool schema."""