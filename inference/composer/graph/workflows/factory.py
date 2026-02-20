"""
Factory functions for creating workflow builders and initial states.
These functions are designed to be called by external services that implement the actual logic
for building workflows and creating initial states based on user data and configurations.
"""

from enum import StrEnum


from .base import GraphBuilder
from .ide.builder import IdeGraphBuilder
from .dialog.builder import DialogGraphBuilder


class WorkFlowType(StrEnum):
    IDE = "ide"
    DIALOG = "dialog"


async def get_builder(workflow_type: WorkFlowType, user_id: str) -> GraphBuilder:
    """Factory function to get the appropriate workflow builder based on type."""

    # 1. Get user configuration from shared data layer
    from db import storage  # pylint: disable=import-outside-toplevel

    user_config = await storage.get_service(storage.user_config).get_user_config(
        user_id
    )

    if workflow_type == WorkFlowType.IDE:
        return IdeGraphBuilder(storage, user_config)
    elif workflow_type == WorkFlowType.DIALOG:
        return DialogGraphBuilder(storage, user_config)
    else:
        raise ValueError(f"Unsupported workflow type: {workflow_type}")
