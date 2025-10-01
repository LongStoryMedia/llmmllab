"""
Engineering Agent for dynamic tool generation and orchestration.
Provides core business logic for tool specification, generation, and management.
"""

from typing import Optional, Any, List

from models.intent_analysis import IntentAnalysis
from models.dynamic_tool import DynamicTool
from composer.monitoring.logging import composer_logger
from composer.tools.registry import ToolRegistry
from utils.grammar_generator import get_grammar_for_model


class EngineeringAgent:
    """
    Engineering Agent for dynamic tool generation with grammar-constrained structured output.
    
    Provides core business logic for tool generation, retrieval, and composition.
    Implements the three-tier decision process: Use Existing -> Modify/Compose -> Create New.
    """

    def __init__(self, pipeline_factory):
        """
        Initialize engineering agent.
        
        Args:
            pipeline_factory: Factory for creating structured pipelines
        """
        self.pipeline_factory = pipeline_factory
        self.tool_registry = ToolRegistry()
        self.logger = composer_logger.logger

    async def orchestrate_tools(
        self, 
        intent: IntentAnalysis, 
        user_id: str, 
        user_query: str = ""
    ) -> List[Any]:
        """
        Orchestrate tool selection and generation based on intent analysis.
        
        Args:
            intent: Intent analysis results
            user_id: User identifier
            user_query: Original user query for context
            
        Returns:
            List of available tools (static and dynamic)
        """
        try:
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
                dynamic_tool = await self.generate_or_retrieve_dynamic_tool(
                    user_query, intent, user_id
                )
                if dynamic_tool:
                    tools.append(dynamic_tool)

            self.logger.info(
                "Tool orchestration completed",
                extra={
                    "user_id": user_id,
                    "tool_count": len(tools)
                }
            )

            return tools

        except Exception as e:
            self.logger.error(
                "Engineering agent tool orchestration failed",
                extra={
                    "user_id": user_id,
                    "error": str(e)
                }
            )
            return []

    async def generate_or_retrieve_dynamic_tool(
        self, 
        user_query: str,
        intent: IntentAnalysis, 
        user_id: str
    ) -> Optional[Any]:
        """
        Generate or retrieve dynamic tool based on requirements.
        
        Args:
            user_query: User's original query
            intent: Intent analysis results
            user_id: User identifier
            
        Returns:
            Dynamic tool instance or None
        """
        try:
            # Create tool specification using Engineering Agent
            tool_spec = await self.generate_tool_specification(
                user_query, intent, user_id
            )
            
            if not tool_spec:
                return None

            # Use ToolRegistry's public dynamic tool generation method
            dynamic_tool = await self.tool_registry.generate_dynamic_tool(
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

    async def generate_tool_specification(
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