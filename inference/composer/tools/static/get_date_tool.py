"""
Static date tool using LangGraph Command pattern.

This tool retrieves the current date and time using ToolRuntime pattern
for proper LangGraph integration.
"""

import json
from datetime import datetime

from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command


@tool
async def get_current_date(
    tool_runtime: ToolRuntime,
) -> Command:
    """
    Get the current date in ISO format.

    This tool returns the current date and time for reference when you need
    to know the current time context for your response.

    Returns:
        Command with current date information
    """
    # Access state and tool_call_id through runtime
    tool_call_id = tool_runtime.tool_call_id
    
    current_date = datetime.now().isoformat()
    
    # Create JSON response message
    response_message = json.dumps({
        "status": "success",
        "current_date": current_date,
        "message": f"Current date and time: {current_date}"
    }, indent=2)
    
    return Command(
        update={
            "current_date": current_date,
            "messages": [ToolMessage(response_message, tool_call_id=tool_call_id)]
        }
    )
