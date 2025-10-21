#!/usr/bin/env python3
"""
Simple test to reproduce the wrapper function issue.
"""
import asyncio
from composer.tools.utils.schema_filter import patch_tool_schema
from composer.tools.static.web_search_tool import web_search

async def test_wrapper_issue():
    print('🧪 Testing wrapper function with different inputs...')
    
    # Test direct tool call (this works)
    print('\n1. Testing direct tool call:')
    try:
        result = await web_search.ainvoke({'query': 'test search'})
        print(f'✅ Direct call works: {type(result)}')
    except Exception as e:
        print(f'❌ Direct call failed: {e}')
    
    # Test patched tool call (this should work)
    print('\n2. Testing patched tool call:')
    patched_tool = patch_tool_schema(web_search)
    try:
        result = await patched_tool.ainvoke({'query': 'test search'})
        print(f'✅ Patched call works: {type(result)}')
    except Exception as e:
        print(f'❌ Patched call failed: {e}')
    
    # Test what happens with empty kwargs (this might be what LangChain is doing)
    print('\n3. Testing patched tool with empty kwargs:')
    try:
        result = await patched_tool.ainvoke({})
        print(f'✅ Empty kwargs works: {type(result)}')
    except Exception as e:
        print(f'❌ Empty kwargs failed: {e}')
    
    # Test what happens with missing query (this is probably the issue)
    print('\n4. Testing patched tool with other kwargs but no query:')
    try:
        result = await patched_tool.ainvoke({'some_other_param': 'value'})
        print(f'✅ Missing query works: {type(result)}')
    except Exception as e:
        print(f'❌ Missing query failed: {e}')

if __name__ == "__main__":
    asyncio.run(test_wrapper_issue())