"""
Test schema filtering fixes with direct ToolsAgentSubgraph execution.
This isolates tool calling functionality from full workflow complexity.
"""

import asyncio
import sys
import os

# Add the inference directory to the path
sys.path.insert(0, "/app")

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from composer.graph.state import WorkflowState
from composer.graph.subgraphs.tools_agent import tools_agent_subgraph

async def test_tools_subgraph_directly():
    """Test ToolsAgentSubgraph directly to verify schema filtering works."""
    
    print("🧪 Testing ToolsAgentSubgraph with schema filtering fixes...")
    
    try:
        # Use the tools subgraph directly (it's already an instance)
        print("✅ Got tools subgraph")
        
        # Create a minimal state with tool calls to trigger tool execution
        initial_state = {
            "messages": [
                HumanMessage(content="Search for latest AI news"),
                AIMessage(
                    content="I'll search for the latest AI news for you.",
                    tool_calls=[{
                        "name": "web_search",
                        "args": {"query": "latest AI news"},
                        "id": "call_test_123"
                    }]
                )
            ],
            "user_id": "test_user_schema_filtering",
            "conversation_id": 123
        }
        
        print("📝 Created initial state with web search request")
        print(f"📧 State messages: {len(initial_state['messages'])}")
        print(f"📧 Message content: {initial_state['messages'][0].content}")
        
        # Execute the tools subgraph
        print("⚡ Executing tools subgraph...")
        
        result_state = await tools_agent_subgraph.graph.ainvoke(initial_state)
        
        print(f"✅ Tools subgraph executed successfully")
        print(f"📊 Result state type: {type(result_state)}")
        print(f"📧 Result messages count: {len(result_state.get('messages', []))}")
        
        # Check for tool calls or responses
        for i, msg in enumerate(result_state.get('messages', [])):
            print(f"   Message {i}: {type(msg).__name__} - {str(msg.content)[:100]}...")
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"      Tool calls: {len(msg.tool_calls)}")
                for j, tc in enumerate(msg.tool_calls):
                    print(f"         Tool call {j}: {tc.get('name', 'unknown')} - {tc.get('id', 'no-id')}")
        
        print("✅ Test completed successfully - no ToolMessage validation errors!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_tools_subgraph_directly())