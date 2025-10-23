#!/usr/bin/env python3
"""
Test script to verify that ToolRuntime validation is working correctly
after the LangGraph subgraph context propagation fix.
"""

import asyncio
from composer.tools.static.web_search_tool import web_search
from composer.tools.static.get_date_tool import get_current_date
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, ToolCall


async def test_tool_runtime_validation():
    """Test that ToolRuntime validation works properly now."""
    print("🧪 Testing ToolRuntime validation fix...")
    
    # Test the date tool (no runtime needed)
    try:
        date_tool_node = ToolNode([get_current_date])
        date_tool_call = ToolCall(
            name='get_current_date',
            args={},
            id='date_call_123'
        )
        
        date_ai_message = AIMessage(content='get date', tool_calls=[date_tool_call])
        date_state = {'messages': [date_ai_message]}
        
        print("\n📅 Testing get_current_date tool...")
        date_result = date_tool_node.invoke(date_state)
        date_message = date_result['messages'][0]
        
        if hasattr(date_message, 'status') and date_message.status == 'error':
            print(f"❌ Date tool error: {date_message.content}")
        else:
            print(f"✅ Date tool success: {date_message.content[:100]}...")
            
    except Exception as e:
        print(f"❌ Date tool exception: {e}")
    
    # Test the web search tool (requires runtime)
    try:
        web_tool_node = ToolNode([web_search])
        web_tool_call = ToolCall(
            name='web_search',
            args={'query': 'test search'},
            id='web_call_123'
        )
        
        web_ai_message = AIMessage(content='search web', tool_calls=[web_tool_call])
        web_state = {
            'messages': [web_ai_message],
            'user_id': 'test_user',
            'user_config': None  # Will use default config
        }
        
        print("\n🔍 Testing web_search tool...")
        web_result = web_tool_node.invoke(web_state)
        web_message = web_result['messages'][0]
        
        if hasattr(web_message, 'status') and web_message.status == 'error':
            print(f"❌ Web search error: {web_message.content}")
            if 'runtime' in web_message.content and 'Field required' in web_message.content:
                print("💥 STILL GETTING TOOLRUNTIME VALIDATION ERROR!")
                return False
            else:
                print("🤔 Different error - ToolRuntime validation might be fixed")
        else:
            print(f"✅ Web search success: {web_message.content[:200]}...")
            return True
            
    except Exception as e:
        print(f"❌ Web search exception: {e}")
        return False
    
    return True


async def main():
    """Main test function."""
    print("🚀 Testing ToolRuntime validation after LangGraph subgraph fix...")
    
    success = await test_tool_runtime_validation()
    
    if success:
        print("\n🎉 ToolRuntime validation appears to be working!")
    else:
        print("\n💥 ToolRuntime validation still has issues")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())