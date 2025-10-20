#!/usr/bin/env python3
"""
Quick test to check tool schema size.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from composer.tools.registry import ToolRegistry

def main():
    """Check tool schema sizes."""
    print("🔧 Checking tool schema sizes...")
    
    # Create tool registry with pipeline factory
    from runner import pipeline_factory
    registry = ToolRegistry(pipeline_factory)
    
    # Get static tools
    static_tools = registry.static_tools
    print(f"Found {len(static_tools)} static tools:")
    
    total_chars = 0
    for tool in static_tools:
        tool_name = getattr(tool, 'name', 'unknown')
        tool_schema = tool.to_openai_tool()
        schema_str = str(tool_schema)
        schema_chars = len(schema_str)
        total_chars += schema_chars
        
        print(f"  {tool_name}: {schema_chars:,} characters")
        if schema_chars > 1000:  # Show content for large schemas
            print(f"    Content preview: {schema_str[:500]}...")
    
    print(f"\nTotal schema characters: {total_chars:,}")
    print(f"Estimated tokens (÷4): {total_chars // 4:,}")
    
    # Also test langchain conversion
    try:
        converted_tools = registry.convert_tools_to_langchain(static_tools)
        print(f"\nLangChain converted tools: {len(converted_tools)}")
        
        # Test the pipeline conversion
        from runner.pipelines.llamacpp.base_llamacpp import BaseLlamaCppPipeline
        
        class TestPipeline(BaseLlamaCppPipeline):
            def __init__(self):
                self.model = type('Model', (), {'name': 'test'})()
                self._logger = print
            
            def _convert_tools_to_openai_format(self, tools):
                return super()._convert_tools_to_openai_format(tools)
            
            def _count_tool_tokens(self, tools):
                return super()._count_tool_tokens(tools)
        
        test_pipeline = TestPipeline()
        openai_tools = test_pipeline._convert_tools_to_openai_format(converted_tools)
        tool_tokens = test_pipeline._count_tool_tokens(openai_tools)
        
        print(f"OpenAI format tools: {len(openai_tools) if openai_tools else 0}")
        print(f"Tool tokens estimated: {tool_tokens:,}")
        
        if openai_tools:
            for i, tool in enumerate(openai_tools[:2]):  # Show first 2
                tool_str = str(tool)
                print(f"  Tool {i+1} size: {len(tool_str):,} characters")
                if len(tool_str) > 1000:
                    print(f"    Preview: {tool_str[:500]}...")
                    
    except Exception as e:
        print(f"Error testing conversion: {e}")

if __name__ == "__main__":
    main()