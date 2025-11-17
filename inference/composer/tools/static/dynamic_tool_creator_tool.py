"""
Dynamic Tool Creator static tool.

This tool lets the agent request creation of a one-off dynamic tool definition.
It returns a JSON spec the system can persist and immediately register.
"""
import json
from typing import List, Optional
from langchain_core.tools import tool

@tool
async def create_dynamic_tool(
    name: str,
    description: str,
    args_schema_json: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> str:
    """Define a new dynamic tool.

    Args:
        name: Unique tool name (snake_case recommended)
        description: Clear purpose; include expected input/output semantics
        args_schema_json: Optional JSON string describing Pydantic args schema fields
        tags: Optional list of classification tags

    Returns:
        JSON specification for dynamic tool registration.
    """
    spec = {
        "name": name.strip(),
        "description": description.strip(),
        "args_schema": args_schema_json or "{}",
        "tags": tags or [],
        "return_direct": False,
        "response_format": "text",
    }
    return json.dumps(spec, ensure_ascii=False)
