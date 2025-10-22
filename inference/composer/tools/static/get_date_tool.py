"""
Static date tool using LangGraph Command pattern.

This tool retrieves the current date and time using ToolRuntime pattern
for proper LangGraph integration.
"""

from datetime import datetime

from langchain_core.tools import tool
from langchain.tools import ToolRuntime


@tool
async def get_current_date(
    runtime: ToolRuntime,
) -> str:
    """
    Get the current date in ISO format.

    This tool returns the current date and time for reference when you need
    to know the current time context for your response.

    Returns:
        Command with current date information
    """
    current_date = datetime.now().isoformat()
    
    # Return date information - ToolNode will automatically create ToolMessage  
    return f"📅 **Current Date and Time**: {current_date}"
