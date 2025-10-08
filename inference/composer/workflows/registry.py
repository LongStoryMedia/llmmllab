"""
Workflow registry - single source of truth for available workflows.
Extracts workflow names from __init__.py exports to ensure consistency.
"""

from typing import Dict, List, Set

# Avoid circular imports by importing directly
from .chat import build_chat_workflow
from .research import build_research_workflow
from .multi_agent import build_multi_agent_workflow
from .creative import build_creative_workflow
from .engineering import build_enhanced_engineering_workflow
from .memory import build_memory_workflow, build_embedding_only_workflow


class WorkflowRegistry:
    """
    Registry of available workflows extracted from __init__.py exports.
    Provides single source of truth for workflow names and validation.
    """

    # Map of workflow builder functions to their workflow names
    _WORKFLOW_BUILDERS = {}

    @classmethod
    def _initialize_builders(cls):
        """Initialize the workflow builders map, filtering out None values."""
        if not cls._WORKFLOW_BUILDERS:  # Only initialize once
            builders = {
                "chat": build_chat_workflow,
                "research": build_research_workflow,
                "multi_agent": build_multi_agent_workflow,
                "creative": build_creative_workflow,
                "engineering": build_enhanced_engineering_workflow,
                "memory": build_memory_workflow,
                "embedding_only": build_embedding_only_workflow,
            }
            # Filter out None values (failed imports)
            cls._WORKFLOW_BUILDERS = {
                k: v for k, v in builders.items() if v is not None
            }

    @classmethod
    def get_available_workflows(cls) -> List[str]:
        """Get list of all available workflow names."""
        cls._initialize_builders()
        return list(cls._WORKFLOW_BUILDERS.keys())

    @classmethod
    def get_workflow_names_set(cls) -> Set[str]:
        """Get set of all available workflow names for fast lookup."""
        cls._initialize_builders()
        return set(cls._WORKFLOW_BUILDERS.keys())

    @classmethod
    def is_valid_workflow(cls, workflow_name: str) -> bool:
        """Check if a workflow name is valid."""
        cls._initialize_builders()
        return workflow_name in cls._WORKFLOW_BUILDERS

    @classmethod
    def get_workflow_builder(cls, workflow_name: str):
        """Get the builder function for a workflow name."""
        cls._initialize_builders()
        return cls._WORKFLOW_BUILDERS.get(workflow_name)

    @classmethod
    def validate_workflows(cls, workflow_names: List[str]) -> List[str]:
        """
        Validate a list of workflow names and return only valid ones.

        Args:
            workflow_names: List of workflow names to validate

        Returns:
            List of valid workflow names
        """
        valid_workflows = []
        valid_set = cls.get_workflow_names_set()

        for name in workflow_names:
            if name in valid_set:
                valid_workflows.append(name)

        return valid_workflows

    @classmethod
    def get_intent_to_workflow_map(cls) -> Dict[str, str]:
        """
        Get mapping from intent keywords to valid workflow names.
        Only includes workflows that actually exist.
        """
        # Only map to workflows that exist
        available = cls.get_workflow_names_set()
        base_map = {
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

        # Filter to only include mappings to available workflows
        return {k: v for k, v in base_map.items() if v in available}

    @classmethod
    def get_workflow_to_subgraph_map(cls) -> Dict[str, str]:
        """
        Get mapping from workflow names to subgraph node names.
        Only includes workflows that actually exist.
        """
        available = cls.get_workflow_names_set()
        return {name: f"{name}_subgraph" for name in available}

    @classmethod
    def get_default_workflow(cls) -> str:
        """Get the default workflow name."""
        return "chat"  # Chat should always be available

    @classmethod
    def get_routing_decision_values(cls) -> List[str]:
        """
        Get values for RoutingDecision enum based on available workflows.
        Includes COORDINATOR as special routing target.
        """
        workflows = cls.get_available_workflows()
        # Add coordinator as special routing decision
        return workflows + ["coordinator"]


# Convenience functions for external use
def get_available_workflows() -> List[str]:
    """Get list of all available workflow names."""
    return WorkflowRegistry.get_available_workflows()


def is_valid_workflow(workflow_name: str) -> bool:
    """Check if a workflow name is valid."""
    return WorkflowRegistry.is_valid_workflow(workflow_name)


def validate_workflows(workflow_names: List[str]) -> List[str]:
    """Validate a list of workflow names and return only valid ones."""
    return WorkflowRegistry.validate_workflows(workflow_names)
