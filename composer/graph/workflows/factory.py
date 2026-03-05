"""
Factory functions for creating workflow builders and initial states.
These functions are designed to be called by external services that implement the actual logic
for building workflows and creating initial states based on user data and configurations.
"""

from enum import StrEnum
from typing import cast


from .base import GraphBuilder
from .ide.builder import IdeGraphBuilder
from .dialog.builder import DialogGraphBuilder
from composer.server.interface import ServerAdapter, ServerInterface


class WorkFlowType(StrEnum):
    IDE = "ide"
    DIALOG = "dialog"


async def get_builder(
    workflow_type: WorkFlowType, user_id: str, user_config
) -> GraphBuilder:
    """Factory function to get the appropriate workflow builder based on type.

    Args:
        workflow_type: Type of workflow to build
        user_id: User identifier
        user_config: UserConfig object (passed from server layer)
    """

    # Create a ServerAdapter to provide server services to the builders
    # This wraps the singleton server and implements ServerInterface
    server_adapter = ServerAdapter()

    # Cast ServerAdapter to ServerInterface since it implements all required attributes
    # At runtime, ServerAdapter wraps the singleton server which has all the services
    server_interface = cast(ServerInterface, server_adapter)

    if workflow_type == WorkFlowType.IDE:
        return IdeGraphBuilder(None, user_config, server_interface=server_interface)
    elif workflow_type == WorkFlowType.DIALOG:
        return DialogGraphBuilder(server_interface, user_config)
    else:
        raise ValueError(f"Unsupported workflow type: {workflow_type}")
