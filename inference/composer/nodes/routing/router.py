"""
Workflow routing node for intelligent workflow selection and execution strategy.
Consolidates workflow-level routing logic from GraphBuilder into a dedicated, reusable component.
"""

from typing import Optional, Dict

from models.workflow_type import WorkflowType
from models.routing_strategy import RoutingStrategy
from composer.graph.state import WorkflowState, RoutingDecision, ExecutionStrategy
from composer.monitoring.logging import composer_logger

# Inline workflow registry to avoid circular imports


class _InlineWorkflowRegistry:
    """Inline workflow registry to avoid import issues."""

    # Available workflows based on composer/workflows/__init__.py exports
    _AVAILABLE_WORKFLOWS = {
        "chat",
        "research",
        "multi_agent",
        "creative",
        "engineering",
        "memory",
        "embedding_only",
    }

    @classmethod
    def is_valid_workflow(cls, workflow_name: str) -> bool:
        """Check if a workflow name is valid."""
        return workflow_name in cls._AVAILABLE_WORKFLOWS

    @classmethod
    def get_available_workflows(cls) -> list:
        """Get list of available workflows."""
        return list(cls._AVAILABLE_WORKFLOWS)

    @classmethod
    def validate_workflows(cls, workflow_names: list) -> list:
        """Validate and return only valid workflows."""
        return [name for name in workflow_names if name in cls._AVAILABLE_WORKFLOWS]

    @classmethod
    def get_default_workflow(cls) -> str:
        """Get default workflow."""
        return "chat"

    @classmethod
    def get_intent_to_workflow_map(cls) -> Dict[str, str]:
        """Get intent to workflow mapping."""
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
            "engineering": "engineering",
            "code": "engineering",
            "debug": "engineering",
            "memory": "memory",
            "remember": "memory",
            "embedding": "embedding_only",
        }

    @classmethod
    def get_workflow_to_subgraph_map(cls) -> Dict[str, str]:
        """Get workflow to subgraph mapping."""
        return {name: f"{name}_subgraph" for name in cls._AVAILABLE_WORKFLOWS}


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
            state.selected_workflows = routing_result.selected_workflows
            state.execution_strategy = ExecutionStrategy(
                routing_result.execution_strategy
            )

            # Set routing decision for deterministic routing if single workflow
            if (
                routing_result.execution_strategy == ExecutionStrategy.SINGLE
                and len(routing_result.selected_workflows) == 1
            ):
                workflow_name = routing_result.selected_workflows[0]
                # Map to RoutingDecision enum if it exists, otherwise use string
                try:
                    state.routing_decision = RoutingDecision(workflow_name)
                except (ValueError, AttributeError):
                    # Fallback if RoutingDecision enum doesn't have this value
                    self.logger.warning(
                        f"Workflow name '{workflow_name}' not in RoutingDecision enum",
                        extra={"user_id": self.user_id, "workflow_name": workflow_name},
                    )
                    state.routing_decision = None

            self.logger.info(
                "Workflow routing determined",
                extra={
                    "user_id": self.user_id,
                    "selected_workflows": routing_result.selected_workflows,
                    "execution_strategy": routing_result.execution_strategy,
                    "routing_reason": routing_result.reason,
                },
            )

            return state

        except Exception as e:
            self.logger.error(
                "Workflow routing failed, falling back to chat",
                extra={"user_id": self.user_id, "error": str(e)},
            )

            # Safe fallback
            state.selected_workflows = ["chat"]
            state.execution_strategy = ExecutionStrategy.SINGLE
            return state

    def determine_routing_strategy(
        self,
        state: WorkflowState,
        explicit_workflow_type: Optional[WorkflowType] = None,
    ) -> RoutingStrategy:
        """
        Determine workflow routing strategy based on state and explicit type.

        Args:
            state: Current workflow state with intent analysis
            explicit_workflow_type: Optional explicit workflow type override

        Returns:
            RoutingStrategy object with routing decision details
        """
        try:
            # Priority 1: Command-based routing (highest priority)
            if hasattr(state, "next_node") and state.next_node:
                # Validate the next_node is a real workflow
                if _InlineWorkflowRegistry.is_valid_workflow(state.next_node):
                    return RoutingStrategy(
                        selected_workflows=[state.next_node],
                        execution_strategy="single",
                        reason="command_based_routing",
                    )
                else:
                    self.logger.warning(
                        f"Invalid workflow in next_node: {state.next_node}",
                        extra={
                            "user_id": self.user_id,
                            "invalid_workflow": state.next_node,
                        },
                    )  # Priority 2: Explicit routing decision from state
            if hasattr(state, "routing_decision") and state.routing_decision:
                workflow_name = (
                    state.routing_decision.value
                    if hasattr(state.routing_decision, "value")
                    else str(state.routing_decision)
                )
                return RoutingStrategy(
                    selected_workflows=[workflow_name],
                    execution_strategy="single",
                    reason="explicit_routing_decision",
                )

            # Priority 3: Explicit workflow type parameter
            if explicit_workflow_type:
                workflow_name = explicit_workflow_type.value
                if _InlineWorkflowRegistry.is_valid_workflow(workflow_name):
                    return RoutingStrategy(
                        selected_workflows=[workflow_name],
                        execution_strategy="single",
                        reason="explicit_workflow_type",
                    )
                else:
                    self.logger.warning(
                        f"Invalid workflow type: {workflow_name}",
                        extra={
                            "user_id": self.user_id,
                            "invalid_workflow": workflow_name,
                        },
                    )

            # Priority 4: Intent-based intelligent routing
            return self._route_by_intent_analysis(state)

        except Exception as e:
            self.logger.error(
                "Error in routing strategy determination",
                extra={"user_id": self.user_id, "error": str(e)},
            )

            # Ultimate fallback
            return RoutingStrategy(
                selected_workflows=["chat"],
                execution_strategy="single",
                reason="error_fallback",
            )

    def _route_by_intent_analysis(self, state: WorkflowState) -> RoutingStrategy:
        """
        Route based on intent analysis and complexity assessment.

        Args:
            state: Workflow state with intent classification

        Returns:
            RoutingStrategy object with routing decision details
        """
        # Extract intent information
        intent = state.intent_classification

        # Check for detailed intent classification
        intent_analysis = getattr(state, "intent_classification", None)
        if intent_analysis and hasattr(intent_analysis, "primary_intent"):
            primary_intent = intent_analysis.primary_intent.lower()
        else:
            primary_intent = str(intent).lower()

        # Use registry to get intent-to-workflow mapping
        intent_map = _InlineWorkflowRegistry.get_intent_to_workflow_map()

        # Try to find matching workflow from intent keywords
        matched_workflow = None
        for intent_keyword, workflow_name in intent_map.items():
            if intent_keyword in primary_intent:
                matched_workflow = workflow_name
                break

        # Complex routing logic with workflow validation
        if complexity == "high" and matched_workflow == "research":
            # High complexity research might benefit from multiple workflows
            valid_workflows = _InlineWorkflowRegistry.validate_workflows(
                ["research", "creative"]
            )
            if len(valid_workflows) >= 2:
                return RoutingStrategy(
                    selected_workflows=valid_workflows,
                    execution_strategy="series",
                    reason="complex_research_series",
                )
            elif valid_workflows:
                return RoutingStrategy(
                    selected_workflows=valid_workflows,
                    execution_strategy="single",
                    reason="research_intent",
                )

        # Single workflow routing based on matched intent
        if matched_workflow and _InlineWorkflowRegistry.is_valid_workflow(
            matched_workflow
        ):
            # Map workflow to appropriate reason
            reason_map = {
                "multi_agent": "multi_agent_intent",
                "creative": "creative_intent",
                "research": "research_intent",
                "engineering": "engineering_intent",
                "memory": "memory_intent",
                "embedding_only": "embedding_intent",
            }
            reason = reason_map.get(matched_workflow, f"{matched_workflow}_intent")

            return RoutingStrategy(
                selected_workflows=[matched_workflow],
                execution_strategy="single",
                reason=reason,
            )

        # Default to chat for unmatched or invalid workflows
        default_workflow = _InlineWorkflowRegistry.get_default_workflow()
        return RoutingStrategy(
            selected_workflows=[default_workflow],
            execution_strategy="single",
            reason="default_chat",
        )

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

            selected_workflows = routing_result.selected_workflows
            execution_strategy = routing_result.execution_strategy

            # For single workflow execution, return the workflow target
            if execution_strategy == "single" and len(selected_workflows) == 1:
                workflow_name = selected_workflows[0]
                # Map workflow names to subgraph node names
                return f"{workflow_name}_subgraph"
            else:
                # For complex execution strategies, route to coordinator
                return "coordinator"

        except Exception as e:
            self.logger.error(
                "Routing target determination failed",
                extra={"user_id": self.user_id, "error": str(e)},
            )
            return "chat_subgraph"  # Safe fallback

    # Routing Maps for Consistency (delegated to _InlineWorkflowRegistry)

    @staticmethod
    def get_intent_to_workflow_map() -> Dict[str, str]:
        """Get consistent mapping from intent keywords to workflow names."""
        return _InlineWorkflowRegistry.get_intent_to_workflow_map()

    @staticmethod
    def get_workflow_to_subgraph_map() -> Dict[str, str]:
        """Get consistent mapping from workflow names to subgraph node names."""
        return _InlineWorkflowRegistry.get_workflow_to_subgraph_map()

    @staticmethod
    def get_available_workflows() -> list:
        """Get list of all available workflow names."""
        return _InlineWorkflowRegistry.get_available_workflows()
