#!/usr/bin/env python3
"""Test tool conversion functionality."""

import sys
import traceback
from composer.tools.registry import ToolRegistry
from runner.pipeline_factory import pipeline_factory

async def test_tool_conversion():
    """Test that Tool model objects can be converted to LangChain tools."""
    print("🧪 Testing tool conversion functionality...")
    
    try:
        # Initialize tool registry
        registry = ToolRegistry(pipeline_factory)
        print("✅ Tool registry initialized")
        
        # Get static tools as Tool model objects
        tool_models = await registry.get_static_tool_instances("test_user")
        print(f"✅ Retrieved {len(tool_models)} tool models")
        
        # Convert to LangChain tools
        langchain_tools = registry.convert_tools_to_langchain(tool_models)
        print(f"✅ Converted to {len(langchain_tools)} LangChain tools")
        
        # Verify the tools are LangChain-compatible
        for i, tool in enumerate(langchain_tools):
            if hasattr(tool, 'name'):
                print(f"  Tool {i+1}: {tool.name} (type: {type(tool).__name__})")
            elif hasattr(tool, '__name__'):
                print(f"  Tool {i+1}: {tool.__name__} (type: {type(tool).__name__})")
            else:
                print(f"  Tool {i+1}: {tool} (type: {type(tool).__name__})")
                
        print("🎉 Tool conversion test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("Traceback:")
        traceback.print_exc()
        return False

async def main():
    """Run the test."""
    success = await test_tool_conversion()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())