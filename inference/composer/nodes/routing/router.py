"""
Workflow routing node for intelligent workflow selection and execution strategy.
Consolidates workflow-level routing logic from GraphBuilder into a dedicated, reusable component.
"""

from typing import Optional, Dict, List, Set

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
        # STRICT: No fallbacks; routing must be derived solely from intent analysis
        routing_result = self.determine_routing_strategy(state)

        state.selected_workflows = routing_result.selected_workflows
        state.execution_strategy = ExecutionStrategy(routing_result.execution_strategy)

        if (
            routing_result.execution_strategy == ExecutionStrategy.SINGLE
            and len(routing_result.selected_workflows) == 1
        ):
            workflow_name = routing_result.selected_workflows[0]
            try:
                state.routing_decision = RoutingDecision(workflow_name)
            except (ValueError, AttributeError):
                # If enum value missing, propagate error (must keep enums aligned)
                raise RuntimeError(
                    f"RoutingDecision enum missing workflow '{workflow_name}'"
                )

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
        # STRICT: ignore explicit overrides; routing is intent-driven only
        if explicit_workflow_type is not None:
            raise RuntimeError(
                "Explicit workflow overrides are disabled in strict mode"
            )
        analyses = getattr(state, "intent_classification", None)
        if not analyses:
            raise RuntimeError("Intent classification unavailable for routing")
        return self._route_by_intent_analysis(state)

    def _route_by_intent_analysis(self, state: WorkflowState) -> RoutingStrategy:
        """
        Route based on intent analysis and complexity assessment.

        Args:
            state: Workflow state with intent classification

        Returns:
            RoutingStrategy object with routing decision details
        """
        # Extract intent information
        # intent_classification is now a list[List[IntentAnalysis]] per updated state
        analyses = getattr(state, "intent_classification", []) or []
        if not isinstance(analyses, list):
            analyses = [analyses]

        # Aggregate intents + complexity to decide workflows
        primary_intents: List[str] = []
        complexities: List[str] = []
        for a in analyses:
            try:
                primary_intents.append(a.primary_intent.lower())
                complexities.append(
                    getattr(
                        a.complexity_level, "value", str(a.complexity_level)
                    ).lower()
                )
            except Exception:  # pragma: no cover - defensive
                continue

        # Fallback if nothing parsed
        if not primary_intents:
            primary_intents = ["chat"]
        if not complexities:
            complexities = ["simple"]

        # Use the most demanding complexity (rough heuristic order)
        complexity_order = ["trivial", "simple", "moderate", "complex", "specialized"]
        complexity_rank = {c: i for i, c in enumerate(complexity_order)}
        complexities = [c for c in complexities if c in complexity_rank]
        selected_complexity = (
            max(complexities, key=lambda c: complexity_rank[c])
            if complexities
            else "simple"
        )

        # Use registry to get intent-to-workflow mapping
        intent_map = _InlineWorkflowRegistry.get_intent_to_workflow_map()

        # Try to find matching workflow from intent keywords
        matched_workflows: Set[str] = set()
        for intent in primary_intents:
            for intent_keyword, workflow_name in intent_map.items():
                if intent_keyword in intent:
                    matched_workflows.add(workflow_name)

        # Engineering heuristic: if any intent contains technical/code keywords route engineering
        if any(
            x in intent
            for intent in primary_intents
            for x in ["code", "debug", "implement", "technical", "refactor"]
        ):
            matched_workflows.add("engineering")

        # Complex routing logic with workflow validation
        if (
            selected_complexity in ["complex", "specialized"]
            and "research" in matched_workflows
        ):
            # Deep research may pair with creative (synthesis) or engineering (if technical)
            candidate = ["research"]
            if "engineering" in matched_workflows:
                candidate.append("engineering")
            else:
                candidate.append("creative")
            valid_workflows = _InlineWorkflowRegistry.validate_workflows(candidate)
            if len(valid_workflows) > 1:
                return RoutingStrategy(
                    selected_workflows=valid_workflows,
                    execution_strategy="series",
                    reason="complex_research_multi_stage",
                    metadata=None,
                )
            elif valid_workflows:
                return RoutingStrategy(
                    selected_workflows=valid_workflows,
                    execution_strategy="single",
                    reason="complex_research_single",
                    metadata=None,
                )

        # Single workflow routing based on matched intent
        if matched_workflows:
            # Filter to valid
            valid = _InlineWorkflowRegistry.validate_workflows(list(matched_workflows))
            if valid:
                # If only one, single strategy else hybrid (series for now)
                if len(valid) == 1:
                    wf = valid[0]
                    reason = f"{wf}_intent"
                    return RoutingStrategy(
                        selected_workflows=valid,
                        execution_strategy="single",
                        reason=reason,
                    )
                else:
                    return RoutingStrategy(
                        selected_workflows=valid,
                        execution_strategy="series",
                        reason="multi_intent_series",
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
        routing_result = self.determine_routing_strategy(state)
        selected_workflows = routing_result.selected_workflows
        execution_strategy = routing_result.execution_strategy
        if execution_strategy == "single" and len(selected_workflows) == 1:
            return f"{selected_workflows[0]}_subgraph"
        return "coordinator"

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
