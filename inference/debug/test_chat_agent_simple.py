#!/usr/bin/env python3
"""
Simple test script to debug ChatAgent tool calling issues.
"""
import asyncio
from composer.tools.utils.schema_filter import patch_tool_schema
from composer.tools.static.web_search_tool import web_search
from composer.graph.state import WorkflowState
from langchain.agents import create_agent
from composer.tools.registry import ToolRegistry

async def test_create_agent_with_tools():
    print('🧪 Testing create_agent with patched tools...')
    
    # Initialize tool registry
    registry = ToolRegistry()
    await registry.initialize()
    tools = registry.get_static_tools()
    print(f'📊 Got {len(tools)} tools: {[tool.name for tool in tools]}')
    
    # Patch tools like ChatAgent does
    filtered_tools = []
    from composer.tools.utils.schema_filter import patch_tool_schema
    for tool in tools:
        filtered_tool = patch_tool_schema(tool)
        filtered_tools.append(filtered_tool)
    print(f'✅ Patched {len(filtered_tools)} tools')
    
    # Try to create agent like ChatAgent does
    try:
        # Use a simple mock LLM for testing
        from langchain_core.language_models.fake import FakeStreamingListLLM
        llm = FakeStreamingListLLM(responses=["I'll search for Python tutorials."])
        
        agent = create_agent(
            model=llm,
            tools=filtered_tools,
            system_prompt='You are a helpful assistant.',
        )
        print('✅ Agent created successfully')
        
        # Try a simple query that should use web search
        messages = [{'role': 'user', 'content': 'Search for Python tutorials'}]
        print('🔄 Testing agent stream...')
        async for chunk in agent.astream({'messages': messages}, stream_mode='messages'):
            print(f'📝 Chunk type: {type(chunk)}')
            # Let's see what happens on first tool call attempt
            if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                print(f'🛠️ Tool calls detected: {chunk.tool_calls}')
            break  # Just test the first chunk
            
    except Exception as e:
        print(f'❌ Agent creation/usage failed: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_create_agent_with_tools())