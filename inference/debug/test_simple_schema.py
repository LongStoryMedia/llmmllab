#!/usr/bin/env python3

"""Simple test to debug the exact tool schema being passed to create_agent"""

import asyncio
import json
from composer.tools.utils.schema_filter import patch_tool_schema

async def test_simple_schema():
    print("🔧 Testing simple schema generation...")
    
    # Create a fake tool with the same signature but simpler implementation
    from langchain_core.tools import BaseTool
    from typing import Annotated
    from pydantic import BaseModel, Field
    
    class WebSearchInput(BaseModel):
        query: str = Field(description="Search query")
    
    class TestWebSearchTool(BaseTool):
        name: str = "web_search"
        description: str = "Search the web for information"
        args_schema: type[BaseModel] = WebSearchInput
        
        def _run(self, query: str) -> str:
            return f"Search results for: {query}"
        
        async def _arun(self, query: str) -> str:
            return f"Search results for: {query}"
    
    # Create the test tool
    test_tool = TestWebSearchTool()
    print(f"Test tool schema: {json.dumps(test_tool.args_schema.model_json_schema(), indent=2)}")
    
    # Apply our filtering
    filtered_tool = patch_tool_schema(test_tool)
    print(f"Filtered tool schema: {json.dumps(filtered_tool.args_schema.model_json_schema(), indent=2)}")
    
    # Test direct invocation
    try:
        result = await filtered_tool.ainvoke({"query": "test"})
        print(f"✅ Direct invoke worked: {result}")
    except Exception as e:
        print(f"❌ Direct invoke failed: {e}")
    
    # Test with kwargs format
    try:
        result = await filtered_tool.ainvoke({"kwargs": {"query": "test"}})
        print(f"✅ Kwargs invoke worked: {result}")
    except Exception as e:
        print(f"❌ Kwargs invoke failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_simple_schema())