"""
Workflow routing node for intelligent workflow selection and execution strategy.
Consolidates workflow-level routing logic from GraphBuilder into a dedicated, reusable component.
"""

from typing import Optional, Dict, Any
from enum import Enum

from models.workflow_type import WorkflowType
from composer.graph.state import WorkflowState, RoutingDecision
from composer.monitoring.logging import composer_logger


class ExecutionStrategy(Enum):
    """Execution strategies for workflow coordination."""
    SINGLE = "single"
    PARALLEL = "parallel" 
    SERIES = "series"
    HYBRID = "hybrid"


class WorkflowRouter:
    """
    Intelligent router for workflow selection and execution strategy.
    
    Consolidates workflow-level routing logic that was previously duplicated
    across GraphBuilder methods. Provides both routing decisions and execution
    strategy determination based on intent analysis and complexity.
    """

    def __init__(self, user_id: str):
        """
        Initialize workflow router.
        
        Args:
            user_id: User identifier for logging and context
        """
        self.user_id = user_id
        self.logger = composer_logger.logger

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Route workflows and determine execution strategy.
        
        This is the main entry point that updates the state with routing decisions
        and execution strategy for the coordinator to use.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with routing decisions
        """
        try:
            routing_result = self.determine_routing_strategy(state)
            
            # Update state with routing decisions
            state.selected_workflows = routing_result["selected_workflows"]
            state.execution_strategy = routing_result["execution_strategy"]
            
            # Set routing decision for deterministic routing if single workflow
            if (routing_result["execution_strategy"] == ExecutionStrategy.SINGLE.value 
                and len(routing_result["selected_workflows"]) == 1):
                workflow_name = routing_result["selected_workflows"][0]
                # Map to RoutingDecision enum if it exists, otherwise use string
                try:
                    state.routing_decision = RoutingDecision(workflow_name)
                except (ValueError, AttributeError):
                    # Fallback if RoutingDecision enum doesn't have this value
                    state.routing_decision = workflow_name

            self.logger.info(
                "Workflow routing determined",
                extra={
                    "user_id": self.user_id,
                    "selected_workflows": routing_result["selected_workflows"],
                    "execution_strategy": routing_result["execution_strategy"],
                    "routing_reason": routing_result.get("reason", "unknown")
                }
            )

            return state

        except Exception as e:
            self.logger.error(
                "Workflow routing failed, falling back to chat",
                extra={"user_id": self.user_id, "error": str(e)}
            )
            
            # Safe fallback
            state.selected_workflows = ["chat"]
            state.execution_strategy = ExecutionStrategy.SINGLE.value
            return state

    def determine_routing_strategy(
        self, 
        state: WorkflowState, 
        explicit_workflow_type: Optional[WorkflowType] = None
    ) -> Dict[str, Any]:
        """
        Determine workflow routing strategy based on state and explicit type.
        
        Args:
            state: Current workflow state with intent analysis
            explicit_workflow_type: Optional explicit workflow type override
            
        Returns:
            Dictionary with routing decision details
        """
        try:
            # Priority 1: Command-based routing (highest priority)
            if hasattr(state, 'next_node') and state.next_node:
                return {
                    "selected_workflows": [state.next_node],
                    "execution_strategy": ExecutionStrategy.SINGLE.value,
                    "reason": "command_based_routing"
                }

            # Priority 2: Explicit routing decision from state
            if hasattr(state, 'routing_decision') and state.routing_decision:
                workflow_name = (state.routing_decision.value 
                               if hasattr(state.routing_decision, 'value') 
                               else str(state.routing_decision))
                return {
                    "selected_workflows": [workflow_name],
                    "execution_strategy": ExecutionStrategy.SINGLE.value,
                    "reason": "explicit_routing_decision"
                }

            # Priority 3: Explicit workflow type parameter
            if explicit_workflow_type:
                return {
                    "selected_workflows": [explicit_workflow_type.value],
                    "execution_strategy": ExecutionStrategy.SINGLE.value,
                    "reason": "explicit_workflow_type"
                }

            # Priority 4: Intent-based intelligent routing
            return self._route_by_intent_analysis(state)

        except Exception as e:
            self.logger.error(
                "Error in routing strategy determination",
                extra={"user_id": self.user_id, "error": str(e)}
            )
            
            # Ultimate fallback
            return {
                "selected_workflows": ["chat"],
                "execution_strategy": ExecutionStrategy.SINGLE.value,
                "reason": "error_fallback"
            }

    def _route_by_intent_analysis(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Route based on intent analysis and complexity assessment.
        
        Args:
            state: Workflow state with intent classification
            
        Returns:
            Routing strategy dictionary
        """
        # Extract intent information
        intent = getattr(state, "intent", "chat")
        complexity = getattr(state, "complexity", "simple")
        
        # Check for detailed intent classification
        intent_analysis = getattr(state, "intent_classification", None)
        if intent_analysis and hasattr(intent_analysis, "primary_intent"):
            primary_intent = intent_analysis.primary_intent.lower()
        else:
            primary_intent = str(intent).lower()

        # Complex routing logic
        if complexity == "high" and ("research" in primary_intent or "analysis" in primary_intent):
            # High complexity research might benefit from multiple workflows
            return {
                "selected_workflows": ["research", "creative"],
                "execution_strategy": ExecutionStrategy.SERIES.value,
                "reason": "complex_research_series"
            }
        elif "multi" in primary_intent or "agent" in primary_intent or "collaboration" in primary_intent:
            return {
                "selected_workflows": ["multi_agent"],
                "execution_strategy": ExecutionStrategy.SINGLE.value,
                "reason": "multi_agent_intent"
            }
        elif "creative" in primary_intent or "generate" in primary_intent or "write" in primary_intent:
            return {
                "selected_workflows": ["creative"],
                "execution_strategy": ExecutionStrategy.SINGLE.value,
                "reason": "creative_intent"
            }
        elif "research" in primary_intent or "analyze" in primary_intent or "search" in primary_intent:
            return {
                "selected_workflows": ["research"],
                "execution_strategy": ExecutionStrategy.SINGLE.value,
                "reason": "research_intent"
            }
        else:
            # Default to chat for simple interactions
            return {
                "selected_workflows": ["chat"],
                "execution_strategy": ExecutionStrategy.SINGLE.value,
                "reason": "default_chat"
            }

    def get_routing_target(self, state: WorkflowState) -> str:
        """
        Get single routing target for conditional edges.
        
        This method provides compatibility with LangGraph conditional routing
        that expects a single string return value.
        
        Args:
            state: Workflow state
            
        Returns:
            Target node name for routing
        """
        try:
            # Use the same logic but return single target
            routing_result = self.determine_routing_strategy(state)
            
            selected_workflows = routing_result["selected_workflows"]
            execution_strategy = routing_result["execution_strategy"]
            
            # For single workflow execution, return the workflow target
            if execution_strategy == ExecutionStrategy.SINGLE.value and len(selected_workflows) == 1:
                workflow_name = selected_workflows[0]
                # Map workflow names to subgraph node names
                return f"{workflow_name}_subgraph"
            else:
                # For complex execution strategies, route to coordinator
                return "coordinator"

        except Exception as e:
            self.logger.error(
                "Routing target determination failed",
                extra={"user_id": self.user_id, "error": str(e)}
            )
            return "chat_subgraph"  # Safe fallback

    # Routing Maps for Consistency
    
    @staticmethod
    def get_intent_to_workflow_map() -> Dict[str, str]:
        """Get consistent mapping from intent keywords to workflow names."""
        return {
            "research": "research",
            "analysis": "research", 
            "analyze": "research",
            "search": "research",
            "creative": "creative",
            "generate": "creative",
            "write": "creative",
            "create": "creative",
            "multi": "multi_agent",
            "agent": "multi_agent",
            "collaboration": "multi_agent",
            "coordinate": "multi_agent",
        }

    @staticmethod
    def get_workflow_to_subgraph_map() -> Dict[str, str]:
        """Get consistent mapping from workflow names to subgraph node names."""
        return {
            "chat": "chat_subgraph",
            "research": "research_subgraph", 
            "creative": "creative_subgraph",
            "multi_agent": "multi_agent_subgraph",
        }