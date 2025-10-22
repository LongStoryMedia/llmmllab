#!/usr/bin/env python3
"""
Test the intelligent tools agent subgraph with enhanced middleware.

This script tests the enhanced architecture with:
1. Fixed ToolRuntime validation errors  
2. Tool call grouping for efficiency (especially web searches)
3. Planning and limiting middleware
4. Sophisticated routing with tool diversity checks
5. Natural agent termination based on completion signals
"""

import asyncio
from models import LangChainMessage, UserConfig
from composer.graph.subgraphs.tools_agent import tools_agent_subgraph
from composer.graph.state import WorkflowState


async def test_intelligent_subgraph():
    """Test the intelligent subgraph with a realistic web search scenario."""
    print("🧪 Testing intelligent tools agent subgraph...")
    
    # Create a test scenario that should trigger multiple web searches
    test_messages = [
        LangChainMessage(
            content="I need to research the latest developments in quantum computing and AI integration. Can you find recent research papers and news about quantum-AI hybrid systems?",
            type="human"
        )
    ]
    
    # Create minimal WorkflowState for testing
    test_state = WorkflowState(
        messages=test_messages,
        user_id="test_user",
        conversation_id=1,
        current_date="2024-10-22",
        available_tools=[],  # Will be populated by tool registry
        user_config=None,  # Minimal test
        web_search_results=[],
        selected_workflows=[]
    )
    
    print(f"📥 Input: {len(test_state.messages)} messages")
    print(f"📤 Query: {test_state.messages[0].content[:100]}...")
    
    try:
        # Execute the intelligent subgraph
        print("\n🔄 Executing intelligent tools agent subgraph...")
        command = await tools_agent_subgraph.execute(test_state)
        
        if command and command.update:
            print(f"✅ Subgraph completed successfully!")
            
            # Apply updates
            for key, value in command.update.items():
                setattr(test_state, key, value)
            
            # Analyze the results  
            print(f"📊 Results:")
            print(f"  - Total messages: {len(test_state.messages)}")
            print(f"  - Web search results: {'Yes' if hasattr(test_state, 'web_search_results') and test_state.web_search_results else 'No'}")
            
            # Count message types
            ai_messages = sum(1 for msg in test_state.messages if hasattr(msg, 'type') and msg.type == 'ai')
            tool_messages = sum(1 for msg in test_state.messages if hasattr(msg, 'type') and msg.type == 'tool')
            
            print(f"  - AI messages: {ai_messages}")
            print(f"  - Tool messages: {tool_messages}")
            
            # Show recent messages
            print(f"\n📝 Recent messages:")
            for i, msg in enumerate(test_state.messages[-5:]):
                msg_type = getattr(msg, 'type', 'unknown')
                content = str(getattr(msg, 'content', ''))[:100]
                tool_calls = getattr(msg, 'tool_calls', None)
                tool_info = f" (has {len(tool_calls)} tool calls)" if tool_calls else ""
                print(f"  {i+1}. [{msg_type}]{tool_info}: {content}...")
                
            # Check for validation errors in tool messages
            error_messages = [msg for msg in test_state.messages if hasattr(msg, 'content') and 'validation error' in str(msg.content).lower()]
            
            if error_messages:
                print(f"❌ VALIDATION ERRORS: Found {len(error_messages)} tool validation errors")
                for err_msg in error_messages[:2]:  # Show first 2 errors
                    print(f"  - {str(err_msg.content)[:100]}...")
            else:
                print("✅ NO VALIDATION ERRORS: All tools executed successfully")
            
            # Check if agent made strategic multiple tool calls
            if ai_messages > 1 and tool_messages > 0:
                print("✅ SUCCESS: Agent demonstrated intelligent multi-tool behavior!")
                
                # Check for tool call grouping efficiency
                web_search_calls = 0
                for msg in test_state.messages:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        web_search_calls += sum(1 for tc in msg.tool_calls if tc.get('name') == 'web_search')
                
                if web_search_calls > 0:
                    efficiency_ratio = tool_messages / web_search_calls
                    print(f"📊 Tool efficiency: {web_search_calls} search calls → {tool_messages} results (ratio: {efficiency_ratio:.2f})")
                    
            else:
                print("⚠️  LIMITED: Agent made fewer tool calls than expected")
                
        else:
            print("❌ Subgraph returned no updates")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_intelligent_subgraph())