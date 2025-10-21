#!/usr/bin/env python3
"""
Test LangChain ToolNode with patched tools.
"""
import asyncio
from composer.tools.utils.schema_filter import patch_tool_schema
from composer.tools.static.web_search_tool import web_search
from langchain_core.messages import AIMessage

async def test_langchain_tool_node():
    print('🧪 Testing LangChain ToolNode with patched tool...')
    
    # Patch web_search tool
    patched_tool = patch_tool_schema(web_search)
    print(f'✅ Tool patched: {patched_tool.name}')
    
    print('🔧 Testing direct tool invoke...')
    try:
        result = await patched_tool.ainvoke({'query': 'test search'})
        print(f'✅ Direct invoke works: {type(result)}')
    except Exception as e:
        print(f'❌ Direct invoke failed: {e}')
        return
    
    # Test what LangChain's ToolNode would do
    print('🔧 Testing LangChain ToolNode behavior...')
    
    from langgraph.prebuilt.tool_node import ToolNode
    
    tool_node = ToolNode([patched_tool])
    
    # Create a message with tool calls like LangChain would
    tool_call = {
        'name': 'web_search', 
        'args': {'query': 'test search'}, 
        'id': 'test_call_123',
        'type': 'tool_call'
    }
    
    ai_message = AIMessage(
        content='I will search for information.',
        tool_calls=[tool_call]
    )
    
    try:
        result = await tool_node.ainvoke({'messages': [ai_message]})
        print(f'✅ ToolNode works: {type(result)}')
        print(f'📝 ToolNode result length: {len(str(result))}')
    except Exception as e:
        print(f'❌ ToolNode failed: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_langchain_tool_node())