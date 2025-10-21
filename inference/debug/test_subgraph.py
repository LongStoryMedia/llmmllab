#!/usr/bin/env python3
"""
Simple test to verify the ToolsAgentSubgraph implementation works correctly.
"""

import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from composer.graph.subgraphs import tools_agent_subgraph


async def test_subgraph():
    """Test the ToolsAgentSubgraph with a simple tool call."""
    
    print("🧪 Testing ToolsAgentSubgraph...")
    
    # Create a simple test state
    test_state = {
        "messages": [
            HumanMessage(content="What's the current date?"),
            AIMessage(
                content="I'll get the current date for you.",
                tool_calls=[{
                    "id": "test_call_1",
                    "name": "get_current_date",
                    "args": {}
                }]
            )
        ],
        "user_id": "test_user",
        "conversation_id": 1,
        "web_search_config": {},
        "memory_config": {},
        "tool_results": []
    }
    
    print("✅ Test state created")
    
    try:
        # Test the subgraph execution
        result = await tools_agent_subgraph.graph.ainvoke(test_state)
        print(f"✅ Subgraph execution completed")
        print(f"   Tool results: {len(result.get('tool_results', []))}")
        
        if result.get("tool_results"):
            for i, tool_result in enumerate(result["tool_results"]):
                print(f"   Result {i+1}: {tool_result.get('status', 'unknown')} - {tool_result.get('name', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Subgraph test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_transform_functions():
    """Test the state transformation functions."""
    
    print("\n🧪 Testing state transformation functions...")
    
    # Mock a minimal WorkflowState
    class MockWorkflowState:
        def __init__(self):
            self.messages = [HumanMessage(content="test")]
            self.user_id = "test_user"
            self.conversation_id = 123
            self.user_config = type('obj', (object,), {
                'web_search': {'max_results': 5},
                'memory': {'enabled': True}
            })()
            self.current_date = "2025-01-01"
            self.things_to_remember = []
    
    mock_state = MockWorkflowState()
    
    try:
        # Test transform to tools state
        tools_state = tools_agent_subgraph.transform_to_tools_state(mock_state)
        print("✅ Transform to ToolsState successful")
        print(f"   User ID: {tools_state['user_id']}")
        print(f"   Conversation ID: {tools_state['conversation_id']}")
        print(f"   Messages count: {len(tools_state['messages'])}")
        
        # Test transform back to main state
        tools_state["tool_results"] = [{"name": "test_tool", "content": "test result", "status": "success"}]
        updates = tools_agent_subgraph.transform_to_main_state(tools_state, mock_state)
        print("✅ Transform to main state successful")
        print(f"   Updates keys: {list(updates.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Transform test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting ToolsAgentSubgraph tests...\n")
    
    success = True
    
    # Test transformation functions first (simpler)
    if not await test_transform_functions():
        success = False
    
    # Test actual subgraph execution
    if not await test_subgraph():
        success = False
    
    if success:
        print("\n✅ All subgraph tests passed!")
    else:
        print("\n❌ Some tests failed")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())