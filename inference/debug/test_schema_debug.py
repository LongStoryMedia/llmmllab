#!/usr/bin/env python3

"""Debug script to check the exact tool schema being passed to create_agent"""

import asyncio
import json
from composer.tools.utils.schema_filter import patch_tool_schema
from composer.tools.registry import ToolRegistry

async def test_schema_debug():
    print("🔧 Testing tool schema filtering and debug...")
    
    # Get web search tool
    from runner import PipelineFactory
    pipeline_factory = PipelineFactory()
    tool_registry = ToolRegistry(pipeline_factory)
    tool_registry.load_static_tools()
    web_search_tool = tool_registry.get_tool('web_search')
    print(f"Original tool: {web_search_tool}")
    
    # Get the original schema
    original_schema = web_search_tool.args_schema.model_json_schema() if hasattr(web_search_tool, 'args_schema') and web_search_tool.args_schema else {}
    print(f"Original schema: {json.dumps(original_schema, indent=2)}")
    
    # Apply schema filtering
    tools = [web_search_tool]
    print(f"Tools before patching: {[tool.name for tool in tools]}")
    
    filtered_tools = patch_tool_schema(tools)
    print(f"Tools after patching: {[tool.name for tool in filtered_tools]}")
    
    # Get the filtered schema
    filtered_web_search = next((tool for tool in filtered_tools if tool.name == 'web_search'), None)
    if filtered_web_search:
        filtered_schema = filtered_web_search.args_schema.model_json_schema() if hasattr(filtered_web_search, 'args_schema') and filtered_web_search.args_schema else {}
        print(f"Filtered schema: {json.dumps(filtered_schema, indent=2)}")
        
        # Test tool directly
        print("\n🧪 Testing filtered tool directly...")
        try:
            # This should work - direct call with query
            result = await filtered_web_search.ainvoke({"query": "test query"})
            print(f"✅ Direct invoke with query worked: {type(result)}")
        except Exception as e:
            print(f"❌ Direct invoke failed: {e}")
            
        try:
            # This is what the LLM is trying - kwargs wrapped
            result = await filtered_web_search.ainvoke({"kwargs": {"query": "test query"}})
            print(f"✅ Direct invoke with kwargs worked: {type(result)}")
        except Exception as e:
            print(f"❌ Direct invoke with kwargs failed: {e}")
    else:
        print("❌ Could not find filtered web_search tool")

if __name__ == "__main__":
    asyncio.run(test_schema_debug())