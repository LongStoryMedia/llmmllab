#!/usr/bin/env python3
"""
Simple debug test to isolate the tool calling issue.
"""

import asyncio
import logging
from composer.tools.utils.schema_filter import patch_tool_schema
from langchain_core.tools import tool

# Set up logging
logging.basicConfig(level=logging.DEBUG)

async def test_tool_issue():
    print("🧪 Testing tool calling issue...")
    
    # Create a proper test function that mimics web_search  
    async def raw_web_search_func(query: str, tool_call_id: str = "test", state=None):
        """A simple test function that mimics web search."""
        return f"Search results for: {query} (tool_call_id: {tool_call_id})"
    
    # Convert function to tool
    @tool
    async def web_search_tool(query: str):
        """Search the web for information."""
        return await raw_web_search_func(query=query, tool_call_id="test", state=None)
    
    # Apply schema filtering
    print("🔧 Applying schema filtering...")
    patched_tool = patch_tool_schema(web_search_tool)
    
    # Print the filtered schema
    print(f"📋 Filtered schema: {patched_tool.args_schema.model_fields}")
    
    # Check if wrapper was applied correctly
    print(f"📋 Tool coroutine after patching: {patched_tool.coroutine}")
    print(f"📋 Tool coroutine type: {type(patched_tool.coroutine)}")
    
    # Test calling the wrapper function directly
    print("📞 Testing wrapper function directly...")
    try:
        # Get the wrapper function that was set as coroutine
        wrapper_func = patched_tool.coroutine
        result = await wrapper_func(query="test search")
        print(f"✅ Wrapper call succeeded: {result}")
    except Exception as e:
        print(f"❌ Wrapper call failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test direct tool call through ainvoke 
    print("📞 Testing tool ainvoke...")
    try:
        result = await patched_tool.ainvoke({"query": "test search"})
        print(f"✅ Tool ainvoke succeeded: {result}")
    except Exception as e:
        print(f"❌ Tool ainvoke failed: {e}")
        import traceback
        traceback.print_exc()

    print("✅ Test completed")

if __name__ == "__main__":
    asyncio.run(test_tool_issue())