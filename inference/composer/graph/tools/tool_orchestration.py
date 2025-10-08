"""
Tool Orchestration Subgraph for dynamic tool generation and management.
Provides sophisticated tool discovery, generation, and orchestration capabilities.
"""

from typing import List, Dict, Any, Optional, Annotated
import operator

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from models import ModelProfileType, PipelinePriority
from models.intent_analysis import IntentAnalysis
from models.dynamic_tool import DynamicTool
from models.tool import Tool
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError
from composer.tools.registry import ToolRegistry
from utils.model_profile import get_model_profile_for_task
from utils.message import extract_message_text
from runner import run_pipeline


class ToolOrchestrationState(BaseModel):
    """State for tool orchestration subgraph with LangGraph reducers."""
    
    model_config = {
        "arbitrary_types_allowed": True,  # Allow LangChain tool types
        "validate_assignment": True,  # Validate on field assignment
        "use_enum_values": True,  # Use enum values in serialization
        "extra": "forbid",  # Prevent extra fields for type safety
    }

    # Input fields
    user_id: str = Field(description="User identifier for configuration")
    user_query: str = Field(description="Original user query for tool generation context")
    intent_analysis: IntentAnalysis = Field(description="Intent analysis results for tool selection")
    
    # Processing state with reducers for proper state management
    static_tools: Annotated[List[Tool], operator.add] = Field(
        default_factory=list, 
        description="Static tools collected from registry"
    )
    dynamic_tools: Annotated[List[Tool], operator.add] = Field(
        default_factory=list, 
        description="Dynamic tools generated from specifications"
    )
    generated_tool_specs: Annotated[List[DynamicTool], operator.add] = Field(
        default_factory=list, 
        description="Generated tool specifications before compilation"
    )
    
    # Output fields with reducers
    orchestrated_tools: Annotated[List[Tool], operator.add] = Field(
        default_factory=list, 
        description="Final orchestrated tool set with deduplication and optimization"
    )
    tool_metadata: Annotated[Dict[str, Any], operator.add] = Field(
        default_factory=dict, 
        description="Metadata about tool orchestration process"
    )
    
    # Error handling and status tracking
    errors: Annotated[List[str], operator.add] = Field(
        default_factory=list, 
        description="Error messages during orchestration process"
    )
    fallback_used: bool = Field(
        default=False, 
        description="Whether fallback mechanisms were used"
    )


class ToolOrchestrationSubgraph:
    """
    Subgraph for sophisticated tool orchestration and dynamic generation.
    
    Implements the complete tool lifecycle:
    1. Static tool retrieval based on intent
    2. Dynamic tool specification generation  
    3. Dynamic tool compilation and validation
    4. Tool deduplication and optimization
    5. Metadata generation for workflow context
    """
    
    def __init__(self, pipeline_factory, tool_registry: Optional[ToolRegistry] = None):
        """
        Initialize tool orchestration subgraph.
        
        Args:
            pipeline_factory: Factory for creating LLM pipelines
            tool_registry: Optional tool registry (creates default if not provided)
        """
        self.pipeline_factory = pipeline_factory
        self.tool_registry = tool_registry or ToolRegistry(pipeline_factory)
        self.logger = composer_logger.logger.bind(component="ToolOrchestrationSubgraph")
    
    def build_subgraph(self) -> CompiledStateGraph:
        """
        Build the tool orchestration subgraph.
        
        Returns:
            Compiled subgraph for tool orchestration
        """
        try:
            # Create subgraph with ToolOrchestrationState
            subgraph = StateGraph(ToolOrchestrationState)
            
            # Add nodes for tool orchestration pipeline
            subgraph.add_node("collect_static_tools", self._collect_static_tools)
            subgraph.add_node("analyze_tool_requirements", self._analyze_tool_requirements)
            subgraph.add_node("generate_dynamic_specs", self._generate_dynamic_tool_specs)
            subgraph.add_node("compile_dynamic_tools", self._compile_dynamic_tools)
            subgraph.add_node("deduplicate_tools", self._deduplicate_and_optimize)
            subgraph.add_node("generate_metadata", self._generate_tool_metadata)
            
            # Define the tool orchestration flow
            subgraph.set_entry_point("collect_static_tools")
            
            subgraph.add_edge("collect_static_tools", "analyze_tool_requirements")
            subgraph.add_edge("analyze_tool_requirements", "generate_dynamic_specs")
            subgraph.add_edge("generate_dynamic_specs", "compile_dynamic_tools") 
            subgraph.add_edge("compile_dynamic_tools", "deduplicate_tools")
            subgraph.add_edge("deduplicate_tools", "generate_metadata")
            subgraph.add_edge("generate_metadata", END)
            
            self.logger.info("Tool orchestration subgraph built successfully")
            return subgraph.compile()
            
        except Exception as e:
            self.logger.error(f"Failed to build tool orchestration subgraph: {e}")
            raise NodeExecutionError(f"Tool orchestration subgraph construction failed: {e}") from e
    
    async def _collect_static_tools(self, state: ToolOrchestrationState) -> ToolOrchestrationState:
        """
        Collect static tools based on intent analysis.
        """
        try:
            self.logger.info(
                "Collecting static tools",
                user_id=state.user_id,
                primary_intent=state.intent_analysis.primary_intent,
            )
            
            # Get static tools from registry based on intent
            static_tools = await self.tool_registry.get_tools_for_context(
                state.intent_analysis, state.user_id
            )
            
            state.static_tools = static_tools
            
            self.logger.info(
                "Static tools collected",
                user_id=state.user_id,
                static_tool_count=len(static_tools),
            )
            
            return state
            
        except Exception as e:
            self.logger.error(f"Static tool collection failed: {e}")
            state.errors = state.errors or []
            state.errors.append(f"Static tool collection failed: {e}")
            state.static_tools = []
            return state
    
    async def _analyze_tool_requirements(self, state: ToolOrchestrationState) -> ToolOrchestrationState:
        """
        Analyze if dynamic tool generation is needed based on intent and available static tools.
        """
        try:
            self.logger.info(
                "Analyzing tool requirements",
                user_id=state.user_id,
                static_tool_count=len(state.static_tools or []),
            )
            
            # Determine if dynamic tools are needed
            needs_dynamic_tools = await self._requires_dynamic_tools(
                state.intent_analysis, state.static_tools or [], state.user_query
            )
            
            # Store analysis in metadata for workflow decision making
            if not state.tool_metadata:
                state.tool_metadata = {}
            
            state.tool_metadata.update({
                "needs_dynamic_tools": needs_dynamic_tools,
                "static_tool_coverage": await self._assess_static_tool_coverage(
                    state.intent_analysis, state.static_tools or []
                ),
                "complexity_requires_generation": self._complexity_requires_generation(state.intent_analysis),
            })
            
            self.logger.info(
                "Tool requirements analyzed", 
                user_id=state.user_id,
                needs_dynamic_tools=needs_dynamic_tools,
            )
            
            return state
            
        except Exception as e:
            self.logger.error(f"Tool requirements analysis failed: {e}")
            state.errors = state.errors or []
            state.errors.append(f"Tool requirements analysis failed: {e}")
            return state
    
    async def _generate_dynamic_tool_specs(self, state: ToolOrchestrationState) -> ToolOrchestrationState:
        """
        Generate dynamic tool specifications using LLM with engineering agent logic.
        """
        try:
            # Skip if dynamic tools not needed
            if not state.tool_metadata.get("needs_dynamic_tools", False):
                self.logger.info("Dynamic tool generation skipped - not required")
                state.generated_tool_specs = []
                return state
            
            self.logger.info(
                "Generating dynamic tool specifications",
                user_id=state.user_id,
                query_length=len(state.user_query),
            )
            
            # Lazy import to avoid circular dependency
            from db import storage  # pylint: disable=import-outside-toplevel
            
            uc = await storage.get_service(storage.user_config).get_user_config(state.user_id)
            # Get engineering model profile for tool generation
            model_profile = await get_model_profile_for_task(
                uc.model_profiles, ModelProfileType.Engineering, state.user_id
            )
            circuit_breaker = model_profile.circuit_breaker or uc.circuit_breaker
            
            # Create tool generation prompt
            prompt = await self._create_tool_generation_prompt(
                state.user_query, state.intent_analysis, state.static_tools or []
            )
            
            # Generate tool specification using engineering model
            with self.pipeline_factory.pipeline(
                model_profile, str, PipelinePriority.NORMAL, circuit_breaker
            ) as pipeline:
                # Use grammar constraint for structured DynamicTool output
                res = await run_pipeline(prompt, pipeline, grammar=DynamicTool)
                tool_spec_text = extract_message_text(res.message) if res and res.message else ""
            
            # Parse the generated tool specification
            tool_specs = await self._parse_tool_specifications(tool_spec_text)
            state.generated_tool_specs = tool_specs
            
            self.logger.info(
                "Dynamic tool specifications generated",
                user_id=state.user_id,
                spec_count=len(tool_specs),
            )
            
            return state
            
        except Exception as e:
            self.logger.error(f"Dynamic tool specification generation failed: {e}")
            state.errors = state.errors or []
            state.errors.append(f"Dynamic tool specification generation failed: {e}")
            state.generated_tool_specs = []
            return state
    
    async def _compile_dynamic_tools(self, state: ToolOrchestrationState) -> ToolOrchestrationState:
        """
        Compile dynamic tool specifications into executable tools.
        """
        try:
            if not state.generated_tool_specs:
                self.logger.info("Dynamic tool compilation skipped - no specifications")
                state.dynamic_tools = []
                return state
            
            self.logger.info(
                "Compiling dynamic tools",
                user_id=state.user_id,
                spec_count=len(state.generated_tool_specs),
            )
            
            dynamic_tools = []
            
            for tool_spec in state.generated_tool_specs:
                try:
                    # Use ToolRegistry to compile tool specification
                    compiled_tool = await self.tool_registry.generate_dynamic_tool(
                        tool_spec, state.user_id
                    )
                    if compiled_tool:
                        dynamic_tools.append(compiled_tool)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to compile tool spec {tool_spec.name}: {e}")
                    # Continue with other tools
            
            state.dynamic_tools = dynamic_tools
            
            self.logger.info(
                "Dynamic tools compiled",
                user_id=state.user_id,
                compiled_count=len(dynamic_tools),
            )
            
            return state
            
        except Exception as e:
            self.logger.error(f"Dynamic tool compilation failed: {e}")
            state.errors = state.errors or []
            state.errors.append(f"Dynamic tool compilation failed: {e}")
            state.dynamic_tools = []
            return state
    
    async def _deduplicate_and_optimize(self, state: ToolOrchestrationState) -> ToolOrchestrationState:
        """
        Deduplicate and optimize the combined tool set.
        """
        try:
            self.logger.info(
                "Deduplicating and optimizing tools",
                user_id=state.user_id,
                static_count=len(state.static_tools or []),
                dynamic_count=len(state.dynamic_tools or []),
            )
            
            # Combine all tools
            all_tools = []
            all_tools.extend(state.static_tools or [])
            all_tools.extend(state.dynamic_tools or [])
            
            # Deduplicate by tool name
            seen_names = set()
            deduplicated_tools = []
            
            for tool in all_tools:
                tool_name = getattr(tool, 'name', str(tool))
                if tool_name not in seen_names:
                    seen_names.add(tool_name)
                    deduplicated_tools.append(tool)
            
            # Optimize tool order (put most relevant tools first based on intent)
            optimized_tools = await self._optimize_tool_order(
                deduplicated_tools, state.intent_analysis
            )
            
            state.orchestrated_tools = optimized_tools
            
            self.logger.info(
                "Tools deduplicated and optimized",
                user_id=state.user_id,
                final_count=len(optimized_tools),
            )
            
            return state
            
        except Exception as e:
            self.logger.error(f"Tool deduplication failed: {e}")
            state.errors = state.errors or []
            state.errors.append(f"Tool deduplication failed: {e}")
            # Fallback to simple combination - ensure type consistency
            combined_tools = []
            combined_tools.extend(state.static_tools or [])
            combined_tools.extend(state.dynamic_tools or [])
            state.orchestrated_tools = combined_tools
            return state
    
    async def _generate_tool_metadata(self, state: ToolOrchestrationState) -> ToolOrchestrationState:
        """
        Generate metadata about the orchestrated tools for workflow context.
        """
        try:
            self.logger.info(
                "Generating tool metadata",
                user_id=state.user_id,
                tool_count=len(state.orchestrated_tools or []),
            )
            
            if not state.tool_metadata:
                state.tool_metadata = {}
            
            # Enhance metadata with orchestration results
            state.tool_metadata.update({
                "total_tools": len(state.orchestrated_tools or []),
                "static_tool_count": len(state.static_tools or []),
                "dynamic_tool_count": len(state.dynamic_tools or []),
                "tool_names": [getattr(tool, 'name', str(tool)) for tool in (state.orchestrated_tools or [])],
                "has_errors": bool(state.errors),
                "error_count": len(state.errors or []),
                "orchestration_success": not bool(state.errors),
            })
            
            self.logger.info(
                "Tool metadata generated",
                user_id=state.user_id,
                metadata_keys=list(state.tool_metadata.keys()),
            )
            
            return state
            
        except Exception as e:
            self.logger.error(f"Tool metadata generation failed: {e}")
            state.errors = state.errors or []
            state.errors.append(f"Tool metadata generation failed: {e}")
            return state
    
    # Helper methods
    
    async def _requires_dynamic_tools(
        self, intent: IntentAnalysis, static_tools: List[Tool], user_query: str
    ) -> bool:
        """Determine if dynamic tool generation is needed."""
        # Check for dynamic/specialized capability requirements
        needs_dynamic = any(
            "DYNAMIC" in str(cap) or "SPECIALIZED" in str(cap)
            for cap in intent.required_capabilities
        )
        
        # Check complexity level
        complex_enough = intent.complexity_level.value in ["COMPLEX", "SPECIALIZED"]
        
        # Check if static tools provide sufficient coverage
        has_sufficient_coverage = len(static_tools) >= 3  # Simple heuristic
        
        # Check if user query indicates need for specialized tools
        query_indicates_specialized = any(
            keyword in user_query.lower() 
            for keyword in ["custom", "specific", "specialized", "unique", "generate"]
        )
        
        return needs_dynamic or (complex_enough and not has_sufficient_coverage) or query_indicates_specialized
    
    async def _assess_static_tool_coverage(
        self, intent: IntentAnalysis, static_tools: List[Tool]
    ) -> float:
        """Assess how well static tools cover the intent requirements."""
        if not intent.required_capabilities or not static_tools:
            return 0.0
        
        # Simple coverage assessment - could be enhanced with semantic matching
        tool_capabilities = set()
        for tool in static_tools:
            if hasattr(tool, 'description'):
                # Extract capabilities from tool descriptions
                tool_capabilities.add(tool.description.lower())
        
        # Calculate rough coverage percentage
        covered_capabilities = 0
        for capability in intent.required_capabilities:
            if any(str(capability).lower() in desc for desc in tool_capabilities):
                covered_capabilities += 1
        
        return covered_capabilities / len(intent.required_capabilities)
    
    def _complexity_requires_generation(self, intent: IntentAnalysis) -> bool:
        """Check if complexity level requires dynamic tool generation."""
        return intent.complexity_level.value in ["COMPLEX", "SPECIALIZED"]
    
    async def _create_tool_generation_prompt(
        self, user_query: str, intent: IntentAnalysis, static_tools: List[Tool]
    ) -> str:
        """Create prompt for dynamic tool generation."""
        
        static_tool_names = [getattr(tool, 'name', str(tool)) for tool in static_tools]
        
        prompt = f"""As a Tool Engineering Specialist, analyze the user's request and generate a dynamic tool specification to address gaps in available capabilities.

User Query: {user_query}
Primary Intent: {intent.primary_intent}
Complexity Level: {intent.complexity_level}
Required Capabilities: {[str(cap) for cap in intent.required_capabilities]}

Available Static Tools: {static_tool_names}

Based on this analysis, create a tool specification that:
1. Addresses specific capability gaps not covered by static tools
2. Is tailored to the user's query and intent
3. Has clear input/output schema definitions
4. Includes proper implementation approach (API calls, calculations, etc.)
5. Considers security and validation requirements

Tool Requirements:
- Must be composable and re-usable
- Should have clear, typed input/output schema
- Must include comprehensive error handling
- Should be efficient and focused on single responsibility
- Must not duplicate existing static tool functionality

Generate a structured tool specification in JSON format matching the DynamicTool schema. Focus on practical implementation that directly addresses the user's needs."""

        return prompt
    
    async def _parse_tool_specifications(self, tool_spec_text: str) -> List[DynamicTool]:
        """Parse LLM-generated tool specifications into DynamicTool objects."""
        try:
            import json
            
            # Try to parse as JSON
            spec_data = json.loads(tool_spec_text)
            
            # Handle both single spec and list of specs
            if isinstance(spec_data, dict):
                spec_data = [spec_data]
            
            tool_specs = []
            for spec in spec_data:
                try:
                    # Create DynamicTool from parsed data
                    tool_spec = DynamicTool(**spec)
                    tool_specs.append(tool_spec)
                except Exception as e:
                    self.logger.warning(f"Failed to parse tool spec: {e}")
            
            return tool_specs
            
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse tool specification as JSON")
            return []
        except Exception as e:
            self.logger.error(f"Tool specification parsing failed: {e}")
            return []
    
    async def _optimize_tool_order(
        self, tools: List[Tool], intent: IntentAnalysis
    ) -> List[Tool]:
        """Optimize tool ordering based on intent relevance."""
        # Simple optimization - put tools with names matching intent keywords first
        intent_keywords = [
            intent.primary_intent.lower(),
            *[str(cap).lower() for cap in intent.required_capabilities]
        ]
        
        relevant_tools = []
        other_tools = []
        
        for tool in tools:
            tool_name = getattr(tool, 'name', '').lower()
            tool_desc = getattr(tool, 'description', '').lower()
            
            is_relevant = any(
                keyword in tool_name or keyword in tool_desc
                for keyword in intent_keywords
            )
            
            if is_relevant:
                relevant_tools.append(tool)
            else:
                other_tools.append(tool)
        
        # Return relevant tools first, then others
        return relevant_tools + other_tools


async def create_tool_orchestration_subgraph(
    pipeline_factory, tool_registry: Optional[ToolRegistry] = None
) -> CompiledStateGraph:
    """
    Factory function to create a tool orchestration subgraph.
    
    Args:
        pipeline_factory: Factory for creating LLM pipelines
        tool_registry: Optional tool registry
        
    Returns:
        Compiled tool orchestration subgraph
    """
    orchestrator = ToolOrchestrationSubgraph(pipeline_factory, tool_registry)
    return orchestrator.build_subgraph()