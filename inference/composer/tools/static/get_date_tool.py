"""
This tool retrieves the current date.
"""

from datetime import datetime
from langchain.tools import tool
from typing import Annotated

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langchain.tools.tool_node import InjectedState
from langgraph.types import Command

from composer.graph.state import WorkflowState


@tool
def get_current_date(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[WorkflowState, InjectedState],
) -> str:
    """
    Get the current date in ISO format.

    This tool returns the current date and time for reference when you need
    to know the current time context for your response.

    Returns:
        str: Current date in ISO format
    """
    return datetime.now().isoformat()
