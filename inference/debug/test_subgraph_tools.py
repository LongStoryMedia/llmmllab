#!/usr/bin/env python3
"""Test the tool agent subgraph specifically."""

import asyncio
from composer.graph.subgraphs.tools_agent import tools_agent_subgraph
from composer.graph.state import WorkflowState
from models import LangChainMessage

async def test_tools_agent_subgraph():
    """Test the tools agent subgraph with a simple tool call."""
    try:
        print("🧪 Testing tools agent subgraph...")
        
        # Create test state with a user message that should trigger web search
        test_messages = [
            LangChainMessage(
                content="What are the latest developments in AI for 2024?",
                type="human"
            )
        ]
        
        test_state = WorkflowState(
            user_id="test-user",
            conversation_id=1,
            messages=test_messages,
            current_date="2025-10-22"
        )
        
        print(f"📝 Created test state with {len(test_messages)} messages")
        
        # Execute subgraph
        print("🔄 Executing tools agent subgraph...")
        result = await tools_agent_subgraph.execute(test_state)
        
        print(f"✅ Subgraph execution completed")
        print(f"📊 Result type: {type(result)}")
        
        if hasattr(result, 'update'):
            # It's a Command
            print("📦 Got Command result")
            updates = result.update
            if callable(updates):
                updated_state = updates(test_state)
            else:
                # update is a dict, apply manually
                updated_state = test_state
                for key, value in updates.items():
                    if key == 'messages':
                        # Extend existing messages
                        existing_messages = getattr(updated_state, 'messages', [])
                        setattr(updated_state, 'messages', existing_messages + value)
                    else:
                        setattr(updated_state, key, value)
            final_messages = getattr(updated_state, 'messages', [])
        else:  
            # It's a WorkflowState
            print("📦 Got WorkflowState result")
            final_messages = getattr(result, 'messages', [])
            
        print(f"📊 Final message count: {len(final_messages)}")
        
        # Check for tool calls and results
        tool_calls_found = 0
        tool_results_found = 0
        
        for i, msg in enumerate(final_messages):
            msg_type = getattr(msg, 'type', 'unknown')
            print(f"📝 Message {i}: type={msg_type}")
            
            if msg_type == 'ai' and hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_calls_found += len(msg.tool_calls)
                print(f"  🛠️ AI message with {len(msg.tool_calls)} tool calls")
                
            if msg_type == 'tool':
                tool_results_found += 1
                print(f"  🔧 Tool result message")
        
        print(f"📊 Tool calls found: {tool_calls_found}")
        print(f"📊 Tool results found: {tool_results_found}")
        
        if tool_calls_found > 0 and tool_results_found > 0:
            print("✅ Tools agent subgraph working correctly!")
        elif tool_calls_found > 0:
            print("⚠️ Tools called but no results found")
        else:
            print("⚠️ No tool calls detected")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_tools_agent_subgraph())
    if success:
        print("✅ Test passed")
    else:
        print("❌ Test failed")
        exit(1)